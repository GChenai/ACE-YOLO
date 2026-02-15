import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\ultralytics-distill\runs\distill\yolov8n-chsim-exp1\weights\best.pt')
    model.val(data=r'D:\ultralytics-distill\dataset\data.yaml',
              split='test',
              imgsz=640,
              batch=1,
            #   save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='RTDETR-r18',
              )