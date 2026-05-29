from pathlib import Path
import time

import cv2
import depthai as dai
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset" / "images"
PREDICT_IMGSZ = 768
PREDICT_CONF = 0.60  # 🎯 提高置信度阈值：只输出模型有60%以上把握的目标，过滤掉瞎猜的误报
PREDICT_IOU = 0.45   # 🎯 稍微收紧重叠抑制，防止同一个接头被画出两个框


def find_latest_weight():
    weights = list((ROOT / "runs" / "detect").glob("*/weights/best.pt"))
    if not weights:
        raise FileNotFoundError("No runs/detect/*/weights/best.pt found. Please train first.")
    return max(weights, key=lambda path: path.stat().st_mtime)


def main():
    weight_path = find_latest_weight()
    print(f"Loading latest YOLOv8 model: {weight_path}", flush=True)
    model = YOLO(str(weight_path))
    print(f"Detecting all trained classes: {model.names}", flush=True)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    saved_count = len(list(DATASET_DIR.glob("*.jpg")))

    print("Initializing OAK camera...", flush=True)
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 640)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    print("Connecting to OAK device...", flush=True)
    with dai.Device(pipeline) as device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        print("====================================")
        print("Camera started. Put the part in front of the lens.")
        print(f"Dataset folder: {DATASET_DIR}")
        print("Press S to save the current raw frame as a dataset image.")
        print("Press Q to quit.")
        print("====================================")

        while True:
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()
            raw_frame = frame.copy()

            results = model.predict(
                source=frame,
                show=False,
                imgsz=PREDICT_IMGSZ,
                conf=PREDICT_CONF,
                iou=PREDICT_IOU,
                classes=[0], # 🎯 这里强制只预测 class 0 (也就是 male 类别)
                verbose=False,
            )
            annotated_frame = results[0].plot()
            cv2.rectangle(annotated_frame, (8, 8), (632, 88), (0, 0, 0), -1)
            cv2.putText(
                annotated_frame,
                "S: save dataset image   Q: quit",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                f"Saved images: {saved_count}  Folder: dataset/images",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (80, 220, 255),
                2,
                cv2.LINE_AA,
            )

            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                print(f"Detected {len(boxes)} object(s)", flush=True)

            cv2.imshow("YOLOv8 Local Live", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                filename = f"img_{int(time.time() * 1000)}.jpg"
                filepath = DATASET_DIR / filename
                cv2.imwrite(str(filepath), raw_frame)
                saved_count += 1
                print(f"Saved dataset image {saved_count}: {filepath}", flush=True)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
