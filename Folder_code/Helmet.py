from ultralytics import YOLO
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def image_detect(model, path):
    results = model.predict(path, conf=0.5)
    im = cv2.imread(path)

    h, w, _ = im.shape

    nohelmet_count = 0
    helmet_count = 0

    # Duyệt từng object
    for box in results[0].boxes:
        cls_id = int(box.cls)
        label = model.names[cls_id].lower()

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Fix out-of-bound
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        if label in ["no helmet", "nohelmet", "no-helmet"]:
            nohelmet_count += 1
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(im, "NO HELMET", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

      
        elif label == "helmet":
            helmet_count += 1
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im, "HELMET", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return im, helmet_count, nohelmet_count


def detect_moto_cropped_save(model, dir_path):
    total_nohelmet = 0

    for filename in os.listdir(dir_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(dir_path, filename)

            img_result, helmet_cnt, nohelmet_cnt = image_detect(model, image_path)

            total_nohelmet += nohelmet_cnt

            print(f"{filename} -> Helmet: {helmet_cnt} | No Helmet: {nohelmet_cnt}")

         
            img_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.title(filename)
            plt.axis("off")
            plt.show()

    print("\n==============================")
    print(f"TỔNG SỐ VI PHẠM (NO HELMET): {total_nohelmet}")


# Load model
model = YOLO("runs\\detect\\helmet_detector6\\weights\\best.pt")

# Chạy
detect_moto_cropped_save(
    model,
    r"D:\Projetc_Helmet\File_anh_test\images"
)
