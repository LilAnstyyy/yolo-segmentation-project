from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    name="my_yolo11_seg_new",
    patience=50
)