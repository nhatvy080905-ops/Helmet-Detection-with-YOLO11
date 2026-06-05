import cv2
from ultralytics import YOLO

# Model 1: detect person + motorbike
model_vehicle = YOLO("yolov8s.pt")

# Model 2: detect helmet
model_helmet = YOLO(r"runs\detect\helmet_detector6\weights\best.pt")

cap = cv2.VideoCapture(0)  # webcam laptop

def is_rider(person_box, bike_box):
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box

    # lấy điểm giữa đáy của người
    foot_x = (px1 + px2) // 2
    foot_y = py2

    return bx1 <= foot_x <= bx2 and by1 <= foot_y <= by2 + 20

while True:
    success, frame = cap.read()
    if not success:
        break

    H, W = frame.shape[:2]

    # detect person + motorcycle
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

    # tìm rider = người ngồi trên xe máy
    riders = []
    for p in persons:
        for b in bikes:
            if is_rider(p, b):
                riders.append(p)
                break

    # kiểm tra helmet cho từng rider
    for (x1, y1, x2, y2) in riders:
        h = y2 - y1
        w = x2 - x1

        # cắt vùng đầu
        hx1 = x1 + int(0.2 * w)
        hx2 = x2 - int(0.2 * w)
        hy1 = y1
        hy2 = y1 + int(0.35 * h)

        # chống out ảnh
        hx1 = max(0, hx1)
        hy1 = max(0, hy1)
        hx2 = min(W, hx2)
        hy2 = min(H, hy2)

        if hx2 <= hx1 or hy2 <= hy1:
            continue

        head = frame[hy1:hy2, hx1:hx2]
        if head.size == 0:
            continue

        # resize head để detect tốt hơn
        head_up = cv2.resize(head, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        results_h = model_helmet(head_up, conf=0.25, verbose=False)[0]

        label = "Unknown"
        best_conf = 0.0

        for box in results_h.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # giả sử class 0 = helmet
            if cls == 0 and conf > 0.5 and conf > best_conf:
                best_conf = conf
                label = "Helmet"

        if label == "Helmet":
            color = (0, 255, 0)
        else:
            color = (0, 255, 255)  # Unknown

        # vẽ box rider
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # vẽ vùng head để debug
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 255, 0), 1)

    cv2.imshow("Helmet Detection Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()