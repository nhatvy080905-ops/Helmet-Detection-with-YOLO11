from ultralytics import YOLO
import cv2
from collections import deque
import numpy as np

# ================== CẤU HÌNH ==================
MODEL_VEHICLE_PATH = "yolov8s.pt"
MODEL_HELMET_PATH = r"runs\detect\helmet_detector6\weights\best.pt"

VIDEO_PATH = r"D:\Projetc_Helmet\Folder_video\helmet5.mp4"
OUTPUT_PATH = "output_optimized.mp4"

CONF_VEHICLE = 0.30
CONF_HELMET = 0.45  # ✅ Tăng từ 0.20 -> 0.45 (giảm false positive)

MIN_HELMET_CONF_TO_ACCEPT = 0.25

SMOOTH_FRAMES = 5  # ✅ Giảm từ 7 -> 5 (nhanh hơn, vẫn smooth)
MAX_MISS = 25  # ✅ Tăng từ 15 -> 25 (60% tăng - tránh mất track)

MIN_MOVEMENT = 8  # ✅ MỚI: Khoảng cách tối thiểu để gọi helmet detect lại
BATCH_SIZE = 8  # ✅ MỚI: Batch detect heads cùng lúc

model_vehicle = YOLO(MODEL_VEHICLE_PATH)
model_helmet = YOLO(MODEL_HELMET_PATH)

print("Helmet model classes:", model_helmet.names)


def normalize_name(name):
    return str(name).lower().replace("-", "_").replace(" ", "_")


def convert_helmet_class(cls_id):
    """Chỉ chuyển class về 2 nhãn: helmet, nohelmet"""
    name = normalize_name(model_helmet.names[int(cls_id)])

    if name in ["helmet", "with_helmet", "wearing_helmet"]:
        return "helmet"

    if name in ["no_helmet", "nohelmet", "without_helmet", "not_helmet"]:
        return "nohelmet"

    return None


def is_rider(person_box, bike_box):
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box

    ph = py2 - py1
    foot_x = (px1 + px2) // 2
    foot_y = py2

    expand_x = int(0.25 * (bx2 - bx1))
    expand_y = int(0.35 * (by2 - by1))

    ebx1 = bx1 - expand_x
    eby1 = by1 - expand_y
    ebx2 = bx2 + expand_x
    eby2 = by2 + expand_y

    foot_inside = ebx1 <= foot_x <= ebx2 and eby1 <= foot_y <= eby2

    lower_person = (px1, py1 + int(0.45 * ph), px2, py2)

    inter_x1 = max(lower_person[0], ebx1)
    inter_y1 = max(lower_person[1], eby1)
    inter_x2 = min(lower_person[2], ebx2)
    inter_y2 = min(lower_person[3], eby2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    lower_area = max(1, (lower_person[2] - lower_person[0]) * (lower_person[3] - lower_person[1]))

    overlap_ratio = inter_area / lower_area

    return foot_inside or overlap_ratio > 0.15


def get_head_region(person_box, frame_w, frame_h):
    x1, y1, x2, y2 = person_box

    w = x2 - x1
    h = y2 - y1

    hx1 = x1 - int(0.18 * w)
    hx2 = x2 + int(0.18 * w)
    hy1 = y1 - int(0.08 * h)
    hy2 = y1 + int(0.38 * h)

    hx1 = max(0, hx1)
    hy1 = max(0, hy1)
    hx2 = min(frame_w, hx2)
    hy2 = min(frame_h, hy2)

    return hx1, hy1, hx2, hy2


def get_optimal_imgsz(head_h, head_w):
    """✅ MỚI: Điều chỉnh imgsz dựa vào kích thước vùng đầu"""
    min_dim = min(head_h, head_w)

    if min_dim < 30:
        return 384
    elif min_dim < 50:
        return 480
    else:
        return 640


def detect_helmet_batch(head_imgs):
    """✅ MỚI: Batch detect - xử lý nhiều vùng đầu cùng lúc"""
    results_batch = []

    for head_img in head_imgs:
        if head_img is None or head_img.size == 0:
            results_batch.append((None, 0.0))
            continue

        h, w = head_img.shape[:2]

        if h < 20 or w < 20:
            results_batch.append((None, 0.0))
            continue

        # ✅ Optimize: Chỉ resize nếu kích thước quá nhỏ (< 40x40)
        if h < 40 and w < 40:
            head_img = cv2.resize(head_img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        elif h < 50 or w < 50:
            head_img = cv2.resize(head_img, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)

        imgsz = get_optimal_imgsz(head_img.shape[0], head_img.shape[1])

        results_h = model_helmet(
            head_img,
            conf=CONF_HELMET,
            iou=0.5,
            imgsz=imgsz,
            verbose=False
        )[0]

        best_label = None
        best_conf = 0.0

        for box in results_h.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = convert_helmet_class(cls)

            if label is None:
                continue

            if conf > best_conf:
                best_conf = conf
                best_label = label

        if best_conf < MIN_HELMET_CONF_TO_ACCEPT:
            results_batch.append((None, best_conf))
        else:
            results_batch.append((best_label, best_conf))

    return results_batch


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def center_distance(box1, box2):
    c1x, c1y = box_center(box1)
    c2x, c2y = box_center(box2)
    return ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) ** 0.5


def majority_vote(history):
    """Lấy kết quả ổn định từ lịch sử frames"""
    valid_labels = [x for x in history if x is not None]

    if len(valid_labels) == 0:
        return None

    helmet_count = valid_labels.count("helmet")
    nohelmet_count = valid_labels.count("nohelmet")

    if helmet_count > nohelmet_count:
        return "helmet"

    if nohelmet_count > helmet_count:
        return "nohelmet"

    return None


def get_dynamic_track_threshold(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    return max(50, 0.35 * max(w, h))


# ================== ĐỌC VIDEO ==================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Không mở được video!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 25

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

if not out.isOpened():
    print("Không tạo được file output!")
    cap.release()
    exit()


tracks = {}
next_track_id = 0


# ================== XỬ LÝ VIDEO ==================
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    H, W = frame.shape[:2]

    # Detect person + motorcycle
    results = model_vehicle(
        frame,
        conf=CONF_VEHICLE,
        iou=0.5,
        imgsz=960,
        classes=[0, 3],
        verbose=False
    )[0]

    persons = []
    bikes = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # DEBUG: In ra tất cả detection
        print(f"Frame {frame_count}: {model_vehicle.names[cls]} (cls={cls}, conf={conf:.2f})")

        if cls == 0:
            persons.append((x1, y1, x2, y2))
            print(f"  → Thêm PERSON: ({x1},{y1})-({x2},{y2})")
        elif cls == 3:
            bikes.append((x1, y1, x2, y2))
            print(f"  → Thêm BIKE: ({x1},{y1})-({x2},{y2})")

    # Ghép người với xe
    riders = []

    for p in persons:
        for b in bikes:
            if is_rider(p, b):
                riders.append(p)
                print(f"  ✓ PERSON ({p[0]},{p[1]}) match BIKE ({b[0]},{b[1]})")
                break

    print(f"Frame {frame_count}: persons={len(persons)}, bikes={len(bikes)}, riders={len(riders)}")

    # Đánh dấu track bị miss
    for tid in list(tracks.keys()):
        tracks[tid]["miss"] += 1

    # ✅ MỚI: Gom head images để batch detect
    batch_heads = []
    batch_track_ids = []
    batch_head_regions = []

    # Xử lý từng rider
    for rider_box in riders:
        x1, y1, x2, y2 = rider_box

        # Gán track ID dựa trên khoảng cách
        assigned_id = None
        min_dist = 999999

        for tid, info in tracks.items():
            dist = center_distance(rider_box, info["box"])
            threshold = get_dynamic_track_threshold(rider_box)

            if dist < min_dist and dist < threshold:
                min_dist = dist
                assigned_id = tid

        if assigned_id is None:
            assigned_id = next_track_id
            next_track_id += 1

            tracks[assigned_id] = {
                "box": rider_box,
                "history": deque(maxlen=SMOOTH_FRAMES),
                "miss": 0,
                "last_label": None,
                "last_conf": 0.0,
                "last_detect_frame": -100  # ✅ MỚI: Lưu frame cuối cùng detect
            }

        tracks[assigned_id]["box"] = rider_box
        tracks[assigned_id]["miss"] = 0

        # ✅ MỚI: Skip helmet detect nếu người chưa di chuyển đủ
        movement = center_distance(rider_box, tracks[assigned_id]["box"]) if "box" in tracks[assigned_id] else 999

        frames_since_detect = frame_count - tracks[assigned_id]["last_detect_frame"]

        should_detect = (
            movement > MIN_MOVEMENT or
            frames_since_detect > SMOOTH_FRAMES or
            tracks[assigned_id]["last_label"] is None
        )

        if should_detect:
            hx1, hy1, hx2, hy2 = get_head_region(rider_box, W, H)
            head = frame[hy1:hy2, hx1:hx2]

            batch_heads.append(head)
            batch_track_ids.append(assigned_id)
            batch_head_regions.append((hx1, hy1, hx2, hy2))
            tracks[assigned_id]["last_detect_frame"] = frame_count
        else:
            # Dùng kết quả cũ nếu không detect
            if tracks[assigned_id]["last_label"] is not None:
                tracks[assigned_id]["history"].append(tracks[assigned_id]["last_label"])

    # ✅ MỚI: Batch detect tất cả heads cùng lúc
    if batch_heads:
        results_batch = detect_helmet_batch(batch_heads)

        for idx, assigned_id in enumerate(batch_track_ids):
            raw_label, raw_conf = results_batch[idx]
            tracks[assigned_id]["history"].append(raw_label)

            final_label = majority_vote(tracks[assigned_id]["history"])

            if final_label is not None:
                tracks[assigned_id]["last_label"] = final_label
                tracks[assigned_id]["last_conf"] = raw_conf
            else:
                if tracks[assigned_id]["last_label"] is not None:
                    final_label = tracks[assigned_id]["last_label"]
                    raw_conf = tracks[assigned_id]["last_conf"]

    # Vẽ kết quả
    for tid, info in list(tracks.items()):
        if info["last_label"] is None:
            continue

        x1, y1, x2, y2 = info["box"]
        final_label = info["last_label"]
        raw_conf = info["last_conf"]

        if final_label == "helmet":
            color = (0, 255, 0)
            text = f"helmet {raw_conf:.2f}"
        elif final_label == "nohelmet":
            color = (0, 0, 255)
            text = f"nohelmet {raw_conf:.2f}"
        else:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"ID:{tid}", (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ✅ Vẽ vùng đầu để debug (nếu cần)
    for idx, (hx1, hy1, hx2, hy2) in enumerate(batch_head_regions):
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 255, 0), 1)

    # Xóa track mất quá lâu
    for tid in list(tracks.keys()):
        if tracks[tid]["miss"] > MAX_MISS:
            del tracks[tid]

    # Ghi frame
    out.write(frame)

    # Hiển thị
    cv2.imshow("Helmet Detection Optimized", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    if frame_count % 100 == 0:
        print(f"Đã xử lý {frame_count} frames, {len(tracks)} tracks đang hoạt động")


cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Đã lưu video ra file: {OUTPUT_PATH}")
