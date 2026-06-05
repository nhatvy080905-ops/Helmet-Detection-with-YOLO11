from ultralytics import YOLO
import cv2
#import matplotlib.pyplot as plt

model = YOLO("runs\\detect\\helmet_detector22\\weights\\best.pt")

image_path = "D:/Projetc_Helmet/File_anh_test/images/anh2.jpg"

results = model.predict(
            image_path, 
            conf=0.15,
            imgsz=960,
            classes = [0,1])

img = cv2.imread(image_path)

names = model.names

colors = {
    "with helmet": (255, 0, 0),
    "without helmet": (0, 0, 255),
}

for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    label = names[cls_id]

    color = colors.get(label, (255, 255, 0))

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def resize_display(img, max_width=600):
    h, w = img.shape[:2]

    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h))

    return img
img = resize_display(img, max_width=600)
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    cls_name = model.names[cls_id]  
    conf = float(box.conf[0])
    if cls_name == "LP":
        continue  # bỏ qua biển số

    print("Phát hiện:", cls_name)
   # print("Độ tin cậy:", conf)
