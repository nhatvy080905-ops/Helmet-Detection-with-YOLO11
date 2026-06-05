import os
import cv2
import mimetypes
import time
import subprocess
import imageio_ffmpeg

from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO(r"D:\Projetc_Helmet\runs\detect\helmet_detector22\weights\best.pt")

FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "1"))
MODEL_CONF = float(os.environ.get("MODEL_CONF", "0.25"))
MODEL_IMGSZ = int(os.environ.get("MODEL_IMGSZ", "960"))


def normalize_name(name):
    return str(name).lower().replace("-", "_").replace(" ", "_")


def get_color(label):
    label = normalize_name(label)

    if label in ["helmet", "with_helmet", "wearing_helmet"]:
        return (255, 0, 0)

    if label in ["nohelmet", "no_helmet", "without_helmet", "not_helmet"]:
        return (0, 0, 255)

    return (255, 255, 0)


def draw_custom_boxes(img, result):
    names = model.names

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        label = names[cls_id]
        color = get_color(label)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        text_y = max(25, y1 - 8)

        cv2.rectangle(img, (x1, text_y - 25), (x1 + 220, text_y + 5), color, -1)

        cv2.putText(
            img,
            text,
            (x1 + 5, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    return img


@app.route("/")
def index():
    return render_template("index.html")
@app.route("/webcam")
def webcam():
    return render_template("webcam.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return "Không tìm thấy file upload", 400

    file = request.files["file"]

    if file.filename == "":
        return "Chưa chọn file", 400

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    ext = file.filename.lower().rsplit(".", 1)[-1]

    if ext in ["jpg", "jpeg", "png"]:
        img = cv2.imread(input_path)

        if img is None:
            return "Không đọc được ảnh", 400

        results = model.predict(
            source=img,
            conf=MODEL_CONF,
            imgsz=MODEL_IMGSZ,
            verbose=False
        )

        img = draw_custom_boxes(img, results[0])

        output_name = "result_" + str(int(time.time())) + ".jpg"
        output_path = os.path.join(RESULT_FOLDER, output_name)

        cv2.imwrite(output_path, img)

        return render_template(
            "index.html",
            result_file=output_name,
            file_type="image"
        )

    elif ext in ["mp4", "avi", "mov", "mkv"]:
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
        out = cv2.VideoWriter(temp_output, fourcc, fps, (output_width, output_height))

        if not out.isOpened():
            cap.release()
            return "Không tạo được video tạm", 500

        frame_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            if frame.shape[1] != output_width or frame.shape[0] != output_height:
                frame = cv2.resize(frame, (output_width, output_height))

            if frame_count % FRAME_SKIP == 0:
                results = model.predict(
                    source=frame,
                    conf=MODEL_CONF,
                    imgsz=MODEL_IMGSZ,
                    verbose=False
                )

                frame = draw_custom_boxes(frame.copy(), results[0])

            out.write(frame)

        cap.release()
        out.release()

        if frame_count == 0:
            return "Video không có frame nào được xử lý", 400

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        try:
            subprocess.run([
                ffmpeg_path,
                "-y",
                "-i", temp_output,
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ], check=True)
        except subprocess.CalledProcessError:
            return "Lỗi khi convert video bằng FFmpeg", 500

        if os.path.exists(temp_output):
            os.remove(temp_output)

        return render_template(
            "index.html",
            result_file=output_name,
            file_type="video"
        )

    else:
        return "File không hợp lệ. Chỉ hỗ trợ ảnh hoặc video.", 400


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
    app.run(debug=True)
