from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO("runs\\detect\\helmet_detector8\\weights\\best.pt")

image_path = "D:/Projetc_Helmet/File_anh_test/images/anh15.jpg"

results = model.predict(image_path, conf=0.5)

img = results[0].plot()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.axis("off")
plt.show()

for box in results[0].boxes:
    cls_id = int(box.cls)
    cls_name = model.names[cls_id]

    if cls_name == "LP":
        continue  # bỏ qua biển số

    print("Phát hiện:", cls_name)