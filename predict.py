
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('./best.pt')

results = model.predict('./images/dog.jpg', save=True, imgsz=320, conf=0.5)

