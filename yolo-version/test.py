from ultralytics import YOLO

model = YOLO("../drowsiness-detection/yolo-version/runs/detect/train/weights/best.pt")
results = model(source=0, show=True)

# model = YOLO("../drowsiness-detection/yolo-version/runs/detect/train/weights/best.pt")
# metrics = model.val(data="datasets.yaml", split='test')
# print(metrics)
