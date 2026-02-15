import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/YOLOv8n/weights/best.pt') # select your model.pt path
    model.predict(source='dataset/VOCdevkit/JPEGImages',
                  project='runs/detect',
                  name='exp',
                  save=True,
                  save_txt=True,
                  #save_conf=True,
                  conf=0.3,
                 # visualize=True # visualize model features maps
                  save_crop = True
                  )