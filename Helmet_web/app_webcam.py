import os
import cv2
import mimetypes
import time
import threading
import subprocess
import imageio_ffmpeg
import numpy as np
from collections import deque

from flask import Flask, Response, render_template, request, send_from_directory, url_for
from ultralytics import YOLO

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True) 
os.makedirs(RESULT_FOLDER, exist_ok=True)

def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def resolve_person_bike_model():
    configured_path = os.environ.get("PERSON_BIKE_MODEL")

    if configured_path:
        return configured_path

    fast_model = os.path.join(PROJECT_DIR, "yolo11m.pt")

    if os.path.exists(fast_model):
        return fast_model

    return os.path.join(PROJECT_DIR, "yolo11m.pt")


model = YOLO(os.environ.get(
    "HELMET_MODEL",
    r"D:\Projetc_Helmet\runs\detect\helmet_detector22\weights\best.pt"
))
model_person_bike = YOLO(resolve_person_bike_model())

PERSON_BIKE_CONF = 0.25
HELMET_CONF = 0.25
PERSON_BIKE_IMGSZ = env_int("PERSON_BIKE_IMGSZ", 640)
HELMET_IMGSZ = env_int("HELMET_IMGSZ", 416)
MIN_HELMET_CONF_TO_ACCEPT = 0.40          # Tăng từ 0.18 → tránh false positive
MAX_PERSON_CANDIDATES = env_int("MAX_PERSON_CANDIDATES", 6)
DEFAULT_CAMERA_WIDTH = env_int("CAMERA_WIDTH", 640)
DEFAULT_CAMERA_HEIGHT = env_int("CAMERA_HEIGHT", 480)
TARGET_INFERENCE_FPS = max(1.0, env_float("INFERENCE_FPS", 8.0))
JPEG_QUALITY = max(40, min(95, env_int("JPEG_QUALITY", 80)))
ENABLE_FULL_FRAME_FALLBACK = os.environ.get("FULL_FRAME_FALLBACK", "1") == "1"  # Bật mặc định
ENABLE_PERSON_FALLBACK = os.environ.get("PERSON_FALLBACK", "1") == "1"

# Temporal voting: số frame lưu lịch sử cho mỗi track
VOTING_WINDOW = env_int("VOTING_WINDOW", 7)
# Dùng ByteTrack để theo dõi người qua các frame
ENABLE_TRACKING = os.environ.get("ENABLE_TRACKING", "1") == "1"

def get_camera_source():
    source = os.environ.get("CAMERA_SOURCE", "0")  

    if source.isdigit():
        return int(source)

    return source


def get_camera_size():
    width = int(os.environ.get("CAMERA_WIDTH", DEFAULT_CAMERA_WIDTH))
    height = int(os.environ.get("CAMERA_HEIGHT", DEFAULT_CAMERA_HEIGHT))

    return width, height


def parse_camera_source(source):
    source = str(source).strip()

    if source.isdigit():
        return int(source)

    return source


def open_camera(camera_source):
    if isinstance(camera_source, int):
        return cv2.VideoCapture(camera_source, cv2.CAP_DSHOW)

    return cv2.VideoCapture(camera_source, cv2.CAP_FFMPEG)


class LatestFrameCapture:
    def __init__(self, camera_source, width, height, is_file_source=False):
        self.camera_source = camera_source
        self.width = width
        self.height = height
        self.is_file_source = is_file_source
        self.cap = open_camera(camera_source)
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.stopped = False
        self.thread = None
        self.read_interval = 0

    def start(self):
        if not self.cap.isOpened():
            return False

        self.configure_capture()

        if self.is_file_source:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 1:
                fps = 30
            self.read_interval = 1.0 / fps

        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

        return True

    def configure_capture(self):
        if isinstance(self.camera_source, int):
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _reader(self):
        while not self.stopped:
            ret, frame = self.cap.read()

            if not ret:
                if self.is_file_source:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                time.sleep(0.02)
                continue

            with self.lock:
                self.frame = frame
                self.frame_id += 1

            if self.read_interval > 0:
                time.sleep(self.read_interval)

    def read_latest(self, last_frame_id=None, timeout=2.0):
        deadline = time.perf_counter() + timeout

        while not self.stopped:
            with self.lock:
                has_new_frame = self.frame is not None and self.frame_id != last_frame_id

                if has_new_frame:
                    return True, self.frame.copy(), self.frame_id

            if time.perf_counter() >= deadline:
                break

            time.sleep(0.005)

        with self.lock:
            if self.frame is not None:
                return True, self.frame.copy(), self.frame_id

        return False, None, last_frame_id

    def release(self):
        self.stopped = True

        if self.thread is not None:
            self.thread.join(timeout=1.0)

        self.cap.release()


def get_request_config():
    source = request.args.get("source", os.environ.get("CAMERA_SOURCE", "0")).strip()
    width = request.args.get("width", os.environ.get("CAMERA_WIDTH", DEFAULT_CAMERA_WIDTH))
    height = request.args.get("height", os.environ.get("CAMERA_HEIGHT", DEFAULT_CAMERA_HEIGHT))

    try:
        width = int(width)
    except ValueError:
        width = DEFAULT_CAMERA_WIDTH

    try:
        height = int(height)
    except ValueError:
        height = DEFAULT_CAMERA_HEIGHT

    return source, width, height


def box_area(box):
    x1, y1, x2, y2 = box

    return max(0, x2 - x1) * max(0, y2 - y1)


def overlap_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    return max(0, x2 - x1) * max(0, y2 - y1)


def is_rider(person_box, bike_box):
    """Kiểm tra người có phải là rider không - điều kiện chặt hơn để tránh false positive."""
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box
    foot_x = (px1 + px2) / 2
    foot_y = py2
    bike_h = by2 - by1
    bike_w = bx2 - bx1

    # Thu hẹp vùng mở rộng: 0.45 → 0.25 để tránh nhầm người đứng gần xe
    expanded_bike = (
        bx1 - int(0.25 * bike_w),
        by1 - int(0.40 * bike_h),
        bx2 + int(0.25 * bike_w),
        by2 + int(0.20 * bike_h)
    )
    foot_inside_bike = (
        expanded_bike[0] <= foot_x <= expanded_bike[2]
        and expanded_bike[1] <= foot_y <= expanded_bike[3]
    )

    lower_person = (
        px1,
        py1 + int(0.55 * (py2 - py1)),  # Chỉ lấy 45% dưới cùng
        px2,
        py2
    )
    lower_overlap = overlap_area(lower_person, expanded_bike)
    lower_overlap_ratio = lower_overlap / max(1, box_area(lower_person))

    # Tăng ngưỡng overlap từ 0.05 → 0.15 để giảm false positive
    # Bỏ điều kiện horizontal+vertical_close vì quá rộng
    return foot_inside_bike or (lower_overlap_ratio > 0.15)


def normalize_helmet_label(label):
    return str(label).lower().strip().replace("-", "_").replace(" ", "_")


def classify_helmet_class(cls_id):
    label = normalize_helmet_label(model.names[int(cls_id)])

    if label in ["without_helmet", "no_helmet", "nohelmet", "not_helmet"]:
        return "no_helmet"

    if "without" in label or "no_helmet" in label or label.startswith("no"):
        return "no_helmet"

    if label in ["with_helmet", "helmet", "wearing_helmet"]:
        return "helmet"

    if "helmet" in label:
        return "helmet"

    return None


def is_helmet_class(cls_id):
    return classify_helmet_class(cls_id) == "helmet"


def clamp_box(box, frame_w, frame_h):
    x1, y1, x2, y2 = box

    x1 = max(0, min(frame_w, x1))
    x2 = max(0, min(frame_w, x2))
    y1 = max(0, min(frame_h, y1))
    y2 = max(0, min(frame_h, y2))

    return x1, y1, x2, y2


def get_head_region(person_box, frame_w, frame_h):
    """Lấy vùng đầu chính xác hơn: chỉ 30% trên cùng thay vì 45%."""
    x1, y1, x2, y2 = person_box
    w = x2 - x1
    h = y2 - y1

    head_box = (
        x1 - int(0.10 * w),
        y1 - int(0.05 * h),
        x2 + int(0.10 * w),
        y1 + int(0.32 * h)   # Giảm từ 0.45 → 0.32: tránh bao gồm vai/ngực
    )

    return clamp_box(head_box, frame_w, frame_h)


def apply_clahe(img):
    """Cải thiện contrast bằng CLAHE trước khi detect - giúp model nhận diện tốt hơn."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def resize_head_for_detection(head_img):
    """Phóng to ảnh đầu + áp dụng CLAHE để tăng chất lượng detect."""
    h, w = head_img.shape[:2]
    min_side = max(1, min(h, w))
    scale = max(1.0, min(2.5, 96 / min_side))  # Giảm max scale 3x → 2.5x

    if scale > 1.0:
        head_img = cv2.resize(
            head_img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

    # Áp dụng CLAHE để tăng độ tương phản
    try:
        head_img = apply_clahe(head_img)
    except Exception:
        pass

    return head_img


def detect_helmet_on_head(head_img):
    if head_img is None or head_img.size == 0:
        return None, 0.0

    h, w = head_img.shape[:2]

    if h < 16 or w < 16:
        return None, 0.0

    head_img = resize_head_for_detection(head_img)
    results_helmet = model.predict(
        head_img,
        conf=HELMET_CONF,
        iou=0.45,
        imgsz=HELMET_IMGSZ,
        verbose=False
    )[0]

    best_label = None
    best_conf = 0.0

    for box in results_helmet.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = classify_helmet_class(cls)

        if label is None:
            continue

        if conf > best_conf:
            best_conf = conf
            best_label = label

    if best_conf < MIN_HELMET_CONF_TO_ACCEPT:
        return None, best_conf

    return best_label, best_conf


def draw_labeled_box(frame, box, label, color, thickness=3):
    x1, y1, x2, y2 = box

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        frame,
        label,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )


def draw_custom_boxes(img, result):
    names = model.names

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        label = names[cls_id]
        helmet_label = classify_helmet_class(cls_id)
        color = (255, 0, 0) if helmet_label == "helmet" else (0, 0, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        text = f"{label} {conf:.2f}"

        cv2.rectangle(img, (x1, y1 - 28), (x1 + 220, y1), color, -1)

        cv2.putText(
            img,
            text,
            (x1 + 5, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
    return img


def draw_full_frame_helmet_detection(frame):
    result = model.predict(
        frame,
        conf=HELMET_CONF,
        iou=0.45,
        imgsz=HELMET_IMGSZ,
        verbose=False
    )[0]
    helmet_count = 0
    no_helmet_count = 0

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        helmet_label = classify_helmet_class(cls_id)

        if helmet_label == "helmet":
            helmet_count += 1
            color = (255, 0, 0)
            label = f"Helmet {conf:.2f}"
        elif helmet_label == "no_helmet":
            no_helmet_count += 1
            color = (0, 0, 255)
            label = f"No Helmet {conf:.2f}"
        else:
            continue

        draw_labeled_box(frame, (x1, y1, x2, y2), label, color)

    cv2.putText(
        frame,
        f"Fallback full frame | Helmet: {helmet_count} | No Helmet: {no_helmet_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return frame


def build_detection_candidates(persons, bikes):
    candidates = []
    used_people = set()

    for idx, person_box in enumerate(persons):
        for bike_box in bikes:
            if is_rider(person_box, bike_box):
                candidates.append((person_box, "Rider"))
                used_people.add(idx)
                break

    if ENABLE_PERSON_FALLBACK:
        for idx, person_box in enumerate(persons):
            if idx in used_people:
                continue
            # Fallback: thêm người khi không có xe hoặc chưa có candidate nào
            if not bikes or not candidates:
                candidates.append((person_box, "Person"))

    return candidates[:MAX_PERSON_CANDIDATES]


# ─── Temporal Voting Buffer ───────────────────────────────────────────────────
# Lưu lịch sử kết quả cho từng track_id để tránh nhãn nhảy loạn giữa các frame
_vote_history: dict[int, deque] = {}
_vote_lock = threading.Lock()


def temporal_vote(track_id: int, label: str | None, conf: float) -> tuple[str | None, float]:
    """
    Lưu kết quả frame hiện tại vào buffer lịch sử và trả về nhãn ổn định
    bằng majority vote trên VOTING_WINDOW frame gần nhất.
    """
    with _vote_lock:
        if track_id not in _vote_history:
            _vote_history[track_id] = deque(maxlen=VOTING_WINDOW)
        _vote_history[track_id].append((label, conf))
        history = list(_vote_history[track_id])

    # Đếm votes
    counts = {"helmet": 0, "no_helmet": 0, None: 0}
    conf_sum = {"helmet": 0.0, "no_helmet": 0.0}
    for lbl, cf in history:
        counts[lbl] = counts.get(lbl, 0) + 1
        if lbl in conf_sum:
            conf_sum[lbl] += cf

    total = len(history)
    helmet_ratio = counts["helmet"] / total
    no_helmet_ratio = counts["no_helmet"] / total

    # Cần ít nhất 40% votes để quyết định (ngưỡng ổn định)
    if helmet_ratio >= 0.40:
        avg_conf = conf_sum["helmet"] / max(1, counts["helmet"])
        return "helmet", avg_conf
    if no_helmet_ratio >= 0.40:
        avg_conf = conf_sum["no_helmet"] / max(1, counts["no_helmet"])
        return "no_helmet", avg_conf
    return None, 0.0


def cleanup_vote_history(active_ids: set):
    """Xoá track_id không còn active để tránh memory leak."""
    with _vote_lock:
        dead = [tid for tid in _vote_history if tid not in active_ids]
        for tid in dead:
            del _vote_history[tid]


def draw_webcam_detection(frame):
    H, W = frame.shape[:2]

    # Dùng track() thay predict() để có track_id ổn định qua các frame
    if ENABLE_TRACKING:
        track_results = model_person_bike.track(
            frame,
            conf=PERSON_BIKE_CONF,
            iou=0.5,
            imgsz=PERSON_BIKE_IMGSZ,
            classes=[0, 3],
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml"
        )
        results = track_results[0] if track_results else None
    else:
        results = model_person_bike(
            frame,
            conf=PERSON_BIKE_CONF,
            iou=0.5,
            imgsz=PERSON_BIKE_IMGSZ,
            classes=[0, 3],
            verbose=False
        )[0]

    persons = []   # list of (box, track_id)
    bikes = []

    if results is not None and results.boxes is not None:
        for box in results.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Lấy track_id nếu có, fallback dùng hash vị trí
            tid = int(box.id[0]) if (box.id is not None and len(box.id) > 0) else hash((x1, y1, x2, y2))
            if cls == 0:
                persons.append(((x1, y1, x2, y2), tid))
            elif cls == 3:
                bikes.append((x1, y1, x2, y2))

    person_boxes = [p[0] for p in persons]
    candidates = build_detection_candidates(person_boxes, bikes)

    # Tạo map box → track_id để ánh xạ sau
    box_to_tid = {p[0]: p[1] for p in persons}

    # Dọn lịch sử vote của track đã mất
    active_tids = {p[1] for p in persons}
    cleanup_vote_history(active_tids)

    if not candidates and ENABLE_FULL_FRAME_FALLBACK:
        return draw_full_frame_helmet_detection(frame)

    if not candidates:
        for x1, y1, x2, y2 in person_boxes:
            draw_labeled_box(frame, (x1, y1, x2, y2), "Person", (120, 120, 120), 2)
        for x1, y1, x2, y2 in bikes:
            draw_labeled_box(frame, (x1, y1, x2, y2), "Motorbike", (0, 180, 255), 2)
        cv2.putText(frame, "Helmet detection waiting...",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    no_helmet_count = 0
    helmet_count = 0
    unknown_count = 0

    for person_box, candidate_type in candidates:
        x1, y1, x2, y2 = person_box
        hx1, hy1, hx2, hy2 = get_head_region(person_box, W, H)

        if hx2 <= hx1 or hy2 <= hy1:
            continue

        head = frame[hy1:hy2, hx1:hx2]
        if head.size == 0:
            continue

        raw_label, raw_conf = detect_helmet_on_head(head)

        # Temporal voting: ổn định nhãn qua VOTING_WINDOW frame
        tid = box_to_tid.get(person_box, hash(person_box))
        helmet_label, best_conf = temporal_vote(tid, raw_label, raw_conf)

        if helmet_label is None:
            unknown_count += 1
            color = (0, 200, 200)
            label = f"{candidate_type} ?"
        elif helmet_label == "helmet":
            helmet_count += 1
            color = (50, 180, 50)   # Xanh lá = an toàn
            label = f"{candidate_type} Helmet {best_conf:.2f}"
        else:
            no_helmet_count += 1
            color = (0, 0, 230)     # Đỏ = nguy hiểm
            label = f"{candidate_type} No Helmet {best_conf:.2f}"

        draw_labeled_box(frame, (x1, y1, x2, y2), label, color)
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 220, 0), 1)

    status = (f"Targets: {len(candidates)} | "
              f"Helmet: {helmet_count} | "
              f"No Helmet: {no_helmet_count} | "
              f"Unknown: {unknown_count}")
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return frame


def create_status_frame(message):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    cv2.putText(
        frame,
        message,
        (40, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    return frame


def encode_stream_frame(frame):
    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )

    if not success:
        return None

    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
    )


def generate_webcam_frames(source=None, width=None, height=None):
    if source is None:
        camera_source = get_camera_source()
    else:
        camera_source = parse_camera_source(source)

    is_file_source = isinstance(camera_source, str) and os.path.isfile(camera_source)

    if width is None or height is None:
        camera_width, camera_height = get_camera_size()
    else:
        camera_width, camera_height = width, height

    capture = LatestFrameCapture(
        camera_source,
        camera_width,
        camera_height,
        is_file_source=is_file_source
    )

    if not capture.start():
        frame = create_status_frame("Cannot open camera")
        encoded = encode_stream_frame(frame)

        if encoded:
            yield encoded

        capture.release()
        return

    last_frame_id = None
    frame_interval = 1.0 / TARGET_INFERENCE_FPS

    try:
        while True:
            loop_started = time.perf_counter()
            ret, frame, last_frame_id = capture.read_latest(last_frame_id)

            if not ret:
                frame = create_status_frame("Cannot read camera frame")
                encoded = encode_stream_frame(frame)

                if encoded:
                    yield encoded

                break

            frame = draw_webcam_detection(frame)
            encoded = encode_stream_frame(frame)

            if encoded:
                yield encoded

            elapsed = time.perf_counter() - loop_started
            sleep_time = frame_interval - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        capture.release()


@app.route("/")
def index():
    source, width, height = get_request_config()
    start = request.args.get("start") == "1"
    stream_url = None

    if start:
        stream_url = url_for(
            "video_feed",
            source=source,
            width=width,
            height=height
        )

    return render_template(
        "webcam.html",
        source=source,
        width=width,
        height=height,
        start=start,
        stream_url=stream_url
    )


@app.route("/webcam")
def webcam():
    return index()


@app.route("/video_feed")
def video_feed():
    source, width, height = get_request_config()

    return Response(
        generate_webcam_frames(source, width, height),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/detect", methods=["POST"])
def detect():
    file = request.files["file"]

    if file.filename == "":
        return "Chưa chọn file"
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    ext = file.filename.lower().split(".")[-1]

    if ext in ["jpg", "jpeg", "png"]:
        results = model.predict(source=input_path, conf=0.15)

        img = cv2.imread(input_path)

        if img is None:
            return "Không đọc được ảnh", 400

        img = draw_custom_boxes(img, results[0])

        output_name = "result_" + str(int(time.time())) + ".jpg"
        output_path = os.path.join(RESULT_FOLDER, output_name)

        cv2.imwrite(output_path, img)

        return render_template(
            "webcam.html",
            result_file=output_name,
            file_type="image"
        )

    elif ext in ["mp4", "avi", "mov"]:
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            return "Không đọc được video đầu vào", 400

        output_name = "result_" + str(int(time.time())) + ".mp4"

        temp_output = os.path.join(RESULT_FOLDER, "temp_" + output_name)
        output_path = os.path.join(RESULT_FOLDER, output_name)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps == 0 or fps is None:
            fps = 25

        output_width = width - (width % 2)
        output_height = height - (height % 2)

        if output_width <= 0 or output_height <= 0:
            cap.release()
            return "Kích thước video không hợp lệ", 400

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            temp_output,
            fourcc,
            fps,
            (output_width, output_height)
        )

        if not out.isOpened():
            cap.release()
            return "Không tạo được video tạm", 500

        frame_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            results = model.predict(frame, conf=0.15)

            annotated_frame = draw_custom_boxes(frame, results[0])

            if annotated_frame.shape[1] != output_width or annotated_frame.shape[0] != output_height:
                annotated_frame = cv2.resize(
                    annotated_frame,
                    (output_width, output_height)
                )

            out.write(annotated_frame)
            frame_count += 1

        cap.release()
        out.release()

        if frame_count == 0:
            return "Video không có frame nào được xử lý", 400

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        subprocess.run([
            ffmpeg_path,
            "-y",
            "-i", temp_output,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path
        ], check=True)

        if os.path.exists(temp_output):
            os.remove(temp_output)

        return render_template(
            "webcam.html",
            result_file=output_name,
            file_type="video"
        )

    else:
        return "File không hợp lệ", 400


@app.route("/results/<filename>")
def result_file(filename):
    file_path = os.path.join(RESULT_FOLDER, filename)

    if not os.path.exists(file_path):
        return "File not found", 404

    mimetype = mimetypes.guess_type(file_path)[0]

    return send_from_directory(
        RESULT_FOLDER,
        filename,
        mimetype=mimetype
    )

if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode, threaded=True, use_reloader=False)
