from ultralytics import YOLO
import cv2
import numpy as np
model_person_bike = YOLO("yolo11s.pt")
model_helmet = YOLO(r"D:\Projetc_Helmet\runs\detect\helmet_detector22\weights\best.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
def is_rider(person_box, bike_box):
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box
    foot_x = (px1 + px2) / 2
    foot_y = py2
    return bx1 <= foot_x <= bx2 and by1 <= foot_y <= by2 + 20

print("Starting webcam feed. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
    H, W = frame.shape[:2]
    results = model_person_bike(frame, conf=0.5, verbose=False)[0]
    persons = []
    bikes = []
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 0:
            persons.append((x1, y1, x2, y2))
        elif cls == 3:
            bikes.append((x1, y1, x2, y2))

    riders  = []
    for p in persons:
        for b in bikes:
            if is_rider(p, b):
                riders.append((p, b))
                break
    
    no_helmet_count = 0
    helmet_count = 0
    for person_box, bike_box in riders:
        x1, y1, x2, y2 = person_box

        h = y2 - y1
        w = x2 - x1

        hx1 = x1+ int(0.2 * w)
        hx2 = x2 - int(0.2 * w)
        hy1 = y1
        hy2 = y1 + int(0.5 * h)

        hx1 = max(0, hx1)
        hx2 = min(W, hx2)
        hy1 = max(0, hy1)
        hy2 = min(H, hy2)

        if hx2 <= hx1 or hy2 <= hy1:
            continue
        head = frame[hy1:hy2, hx1:hx2]
        if head.size == 0:
            continue

        head_resized = cv2.resize(head, None, fx = 2.0, fy = 2.0, interpolation = cv2.INTER_CUBIC)

        results_helmet = model_helmet(head_resized, conf=0.5, verbose=False)[0]
        has_helmet = False
        best_conf = 0.0
        for box in results_helmet.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if conf > best_conf:
                best_conf = conf
                if cls == 0: 
                    has_helmet = True

        if has_helmet:
            helmet_count += 1
            color = (255, 0, 0)
            label = f"Helmet {best_conf:.2f}"
        else:
            no_helmet_count += 1
            color = (0, 0, 255)
            label = f"No Helmet {best_conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 255, 0), 1)
    cv2.putText(frame, f"Riders: {len(riders)} | With Helmet: {helmet_count} | No Helmet: {no_helmet_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Helmet Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()     
print("Webcam feed ended.")
