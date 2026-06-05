from ultralytics import YOLO
import cv2

# Load model
model_vehicle = YOLO("yolov8s.pt")
model_helmet = YOLO(r"runs\detect\helmet_detector6\weights\best.pt")

# Mở video đầu vào
cap = cv2.VideoCapture("helmet9.mp4")

if not cap.isOpened():
    print("Không mở được video!")
    exit()

# Lấy thông tin video
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25  # fallback nếu OpenCV không đọc được FPS

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Tạo file video đầu ra
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

if not out.isOpened():
    print("Không tạo được file output.mp4!")
    cap.release()
    exit()


def is_rider(person_box, bike_box): 
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box

    # lấy điểm chân người
    foot_x = (px1 + px2) // 2
    foot_y = py2

    return bx1 <= foot_x <= bx2 and by1 <= foot_y <= by2 + 20


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]

    # Detect person + motorcycle
    results = model_vehicle(frame, conf=0.4, verbose=False)[0]

    persons = []
    bikes = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if conf < 0.4:
            continue

        if cls == 0:   # person
            persons.append((x1, y1, x2, y2))
        elif cls == 3: # motorcycle
            bikes.append((x1, y1, x2, y2))

    # Ghép người lái với xe
    riders = []
    for p in persons:
        for b in bikes:
            if is_rider(p, b):
                riders.append(p)
                break

    # Detect helmet trên vùng đầu
    for (x1, y1, x2, y2) in riders:
        h = y2 - y1
        w = x2 - x1

        # cắt vùng đầu tương đối
        hx1 = x1 + int(0.2 * w)
        hx2 = x2 - int(0.2 * w)
        hy1 = y1
        hy2 = y1 + int(0.35 * h)

        # ép biên
        hx1 = max(0, hx1)
        hy1 = max(0, hy1)
        hx2 = min(W, hx2)
        hy2 = min(H, hy2)

        if hx2 <= hx1 or hy2 <= hy1:
            continue

        head = frame[hy1:hy2, hx1:hx2]
        if head.size == 0:
            continue

        results_h = model_helmet(head, conf=0.4, verbose=False)[0]

        label = "No Helmet"
        best_conf = 0.0

        for box in results_h.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # giả sử class 0 = Helmet
            if cls == 0 and conf > 0.5 and conf > best_conf:
                best_conf = conf
                label = "Helmet"

        color = (0, 255, 0) if label == "Helmet" else (0, 0, 255)

        # vẽ box người
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # vẽ box vùng đầu để debug
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 255, 0), 1)

    # GHI FRAME RA VIDEO
    out.write(frame)

    # HIỂN THỊ VIDEO
    cv2.imshow("Helmet Detection", frame)

    # Nhấn ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Giải phóng
cap.release()
out.release()
cv2.destroyAllWindows()

print("Đã lưu video ra file output.mp4")