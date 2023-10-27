
from ultralytics import YOLO

# 加载模型
model = YOLO('./best.pt')

# 要推理的图片
results = model.predict('./images/dog.jpg', save=True, imgsz=320, conf=0.5)

