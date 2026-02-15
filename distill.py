import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.distill import DetectionDistiller
from ultralytics.models.yolo.segment.distill import SegmentationDistiller
from ultralytics.models.yolo.pose.distill import PoseDistiller
from ultralytics.models.yolo.obb.distill import OBBDistiller

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': r'D:\ultralytics-distill\runs\prune\yolov8-lamp-exp3-prune2\weights\prune.pt',
        'data':r'D:\ultralytics-distill\dataset\data.yaml',
        'imgsz': 640,
        'epochs': 554,
        'batch': 4,
        'workers': 0,
        'cache': True,
        'optimizer': 'auto',
        'device': '0',
        'close_mosaic': 20,
        # 'amp': False, # 如果蒸馏损失为nan，请把amp设置为False
        'project':'runs/distill',
        'name':'yolov8n-chsim-exp1',
        
        # distill
        'prune_model': True,
        'teacher_weights': r'D:\ultralytics-distill\runs\train\exp\weights\best.pt',
        'teacher_cfg': r'D:\ultralytics-distill\ultralytics\cfg\models\v8\yolov8l.yaml',
        'kd_loss_type': 'feature',
        'kd_loss_decay': 'constant',
        
        'logical_loss_type': 'l2',
        'logical_loss_ratio': 1.0,
        
        'teacher_kd_layers': '12,15,18,21',
        'student_kd_layers': '12,15,18,21',
        'feature_loss_type': 'cwd',
        'feature_loss_ratio': 1.0
    }
    
    model = DetectionDistiller(overrides=param_dict)
    # model = SegmentationDistiller(overrides=param_dict)
    # model = PoseDistiller(overrides=param_dict)
    # model = OBBDistiller(overrides=param_dict)
    model.distill()