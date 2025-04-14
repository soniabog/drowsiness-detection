from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(
    data="datasets.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)
