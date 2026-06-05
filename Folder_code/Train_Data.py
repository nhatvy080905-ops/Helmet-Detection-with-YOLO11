from ultralytics import YOLO

model = YOLO('yolo11n.pt')

# Train
model.train(
    data=r'D:\Projetc_Helmet\Data_new\data.yaml',  # đường dẫn file yaml
    epochs=50,                 # số epoch
    imgsz=640,                 # kích thước ảnh
    batch=16,                  # batch size
    name='helmet_detector',    # tên project
    device='cpu'                 # GPU (Colab)
)
