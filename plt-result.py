import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 投稿级风格设置
# =========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2

pwd = os.getcwd()

names = ['YOLOv8L','YOLOv8n','YOLOv8n-ACE','YOLOv8n-ACE-prune','YOLOv8n-ACE-prune-distill']

# =========================
# 读取数据
# =========================
all_data = {}

for name in names:
    path = f'runs/train/{name}/results.csv'
    data = pd.read_csv(path)

    data.columns = data.columns.str.strip()

    if name != 'YOLOv8L':
        data = data.iloc[:550]

    all_data[name] = data


# =========================
# 清洗函数
# =========================
def clean_column(data, col):
    y = data[col].astype(np.float32).replace(np.inf, np.nan)
    y = y.interpolate()
    return y


# =========================
# 1️⃣ Metrics 曲线（图例右下）
# =========================
plt.figure(figsize=(8, 8))

metrics = [
    ('metrics/precision(B)', 'Precision'),
    ('metrics/recall(B)', 'Recall'),
    ('metrics/mAP50(B)', 'mAP@0.5'),
    ('metrics/mAP50-95(B)', 'mAP@0.5:0.95')
]

for idx, (col, ylabel) in enumerate(metrics):
    plt.subplot(2, 2, idx + 1)

    for name in names:
        y = clean_column(all_data[name], col)
        plt.plot(y, label=name)

    plt.xlabel('Epoch')
    plt.ylabel(ylabel)

    # ❌ 不再使用 plt.title()
    plt.legend(loc='lower right', fontsize=9, frameon=True)
    plt.grid(False)

plt.tight_layout()
metric_path = os.path.join(pwd, 'Figure_metrics.tif')
plt.savefig(metric_path, dpi=600)
plt.close()

print(f'Metrics figure saved to {metric_path}')


# =========================
# 2️⃣ Loss 曲线（图例右上）
# =========================
plt.figure(figsize=(10, 6))

losses = [
    ('train/box_loss', 'Train Box Loss'),
    ('train/dfl_loss', 'Train DFL Loss'),
    ('train/cls_loss', 'Train CLS Loss'),
    ('val/box_loss', 'Val Box Loss'),
    ('val/dfl_loss', 'Val DFL Loss'),
    ('val/cls_loss', 'Val CLS Loss')
]

for idx, (col, ylabel) in enumerate(losses):
    plt.subplot(2, 3, idx + 1)

    for name in names:
        y = clean_column(all_data[name], col)
        plt.plot(y, label=name)

    plt.xlabel('Epoch')
    plt.ylabel(ylabel)

    # ❌ 不再使用 plt.title()
    plt.legend(loc='upper right', fontsize=9, frameon=True)
    plt.grid(False)

plt.tight_layout()
loss_path = os.path.join(pwd, 'Figure_loss.tif')
plt.savefig(loss_path, dpi=600)
plt.close()

print(f'Loss figure saved to {loss_path}')
print("All figures generated successfully.")