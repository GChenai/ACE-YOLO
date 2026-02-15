import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v5/yolov5n.yaml')
    #model.load(r'D:\ultralytics-main\yolov8n.pt') # loading pretrain weights
    model.train(data=r'D:\ultralytics-distill\dataset\data.yaml',
                cache=False,
                imgsz=640,
                epochs=100000,
                batch=4,
                workers=0,
                device='0',
                # resume='', # last.pt path
                project='runs/train',
                name='exp',
                patience=30,
                )