from ultralytics import YOLO

if __name__ == "__main__":
    # 1. 加载一个超轻量级的 YOLOv8n 预训练模型
    model = YOLO("yolov8n.pt")
    
    # 2. 开始在本地进行训练
    # data: 指向我们刚才生成的配置文件
    # epochs: 训练轮数 (由于这里只是测试代码或者数据很少，可以设置50到100)
    # imgsz: 图片大小，640 是最标准的
    print("🚀 开始在本地环境训练专属模型...")
    results = model.train(
        data="E:/shijue/yolo_dataset/data.yaml", 
        epochs=150, # 提升训练轮数到150，加强模型学习
        imgsz=640,
        batch=2, # 对于显存/内存小的电脑，batch 调小不容易崩溃
        device='cpu' # 如果你只有核显或电脑没配好CUDA，用cpu训练这几张图也很快
    )
    
    print("\n🎉 训练完成！模型保存在 runs/detect/train/weights 中。")
