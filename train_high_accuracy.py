from ultralytics import YOLO
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def choose_model():
    previous_best = ROOT / "runs" / "detect" / "head_gpu_high_accuracy" / "weights" / "best.pt"
    if previous_best.exists():
        return str(previous_best)

    previous_best_2 = ROOT / "runs" / "detect" / "train-2" / "weights" / "best.pt"
    if previous_best_2.exists():
        return str(previous_best_2)

    return str(ROOT / "yolov8n.pt")


if __name__ == "__main__":
    model_path = choose_model()
    print(f"Using model: {model_path}")
    model = YOLO(model_path)

    results = model.train(
        data="E:/shijue/yolo_dataset/data.yaml",
        epochs=260,
        imgsz=768,
        batch=16,
        device="cpu",
        workers=4,
        amp=False,
        patience=60,
        cos_lr=True,
        close_mosaic=25,
        project="E:/shijue/runs/detect",
        name="train-3",
        exist_ok=True,
    )

    print("High-accuracy training complete (V3).")
    print(results)
