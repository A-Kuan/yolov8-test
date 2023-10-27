from pathlib import Path
from ultralytics import YOLO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if __name__ == '__main__':


    # Create a new YOLO model from scratch
    model = YOLO('yolov8n.yaml')

    # # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')

    # # Train the model using the 'coco128.yaml' dataset for 3 epochs
    # 数据集配置文件
    results = model.train(data=str(ROOT) + '/datasets/animal/data.yaml', epochs=100, imgsz=640)


    # Evaluate the model's performance on the validation set
    results = model.val()

    # # Perform object detection on an image using the model
    results = model('./cat.jpg')

    success = model.export()
