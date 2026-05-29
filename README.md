# NJIT Robocon 视觉抓取项目 (OAK + YOLOv8)

本项目是一个基于 OAK (OpenCV AI Kit) 深度相机和 YOLOv8 目标检测模型的视觉抓取系统。它可以实现物体的精确识别和 3D 空间坐标定位，用于机械臂的引导抓取。

## 目录结构
- `grasp_yolo_hsv_fusion.py`: 核心视觉抓取代码（YOLO和HSV颜色空间融合识别与定位）
- `train_high_accuracy.py`: 模型高精度训练脚本
- `train_local.py`: 本地模型训练脚本
- `check_oak.py`: 相机连接检查与基础测试 
- `live_oak_yolo.py`: 实时 YOLO 检测与 OAK 测试

## 运行环境
- Python 开发环境
- 主要依赖：`depthai`, `ultralytics`, `opencv-python`, `numpy`

## 快速开始

1. **配置环境**
   ```bash
   pip install -r requirements.txt
   ```
   *(如果需要 GPU 加速请先配置好 PyTorch 环境)*

2. **运行模型**
   ```bash
   python grasp_yolo_hsv_fusion.py
   ```
   
## 注意事项
模型权重文件（`.pt`）和数据集很大，已在 `.gitignore` 中被忽略。使用前请确保本地具有对应的模型文件（如 `yolov8n.pt` 等）。