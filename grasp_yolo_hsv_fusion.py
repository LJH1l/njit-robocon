import cv2
import numpy as np
import depthai as dai
from ultralytics import YOLO
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
PREDICT_IMGSZ = 768
PREDICT_CONF = 0.50
PREDICT_IOU = 0.45

def find_latest_weight():
    weights = list((ROOT / "runs" / "detect").glob("*/weights/best.pt"))
    if not weights:
        return ROOT / "yolov8n.pt"
    # 根据文件的最后修改时间，自动选择最新生成的模型（比如刚训练出来的 train-3）
    return max(weights, key=lambda path: path.stat().st_mtime)

def main():
    weight_path = find_latest_weight()
    print(f"Loading YOLO model: {weight_path}")
    model = YOLO(str(weight_path))
    
    # ---------------- 相机管线设置 ----------------
    pipeline = dai.Pipeline()
    
    camRgb = pipeline.create(dai.node.ColorCamera)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    spatialCalc = pipeline.create(dai.node.SpatialLocationCalculator)
    
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutSpatialData = pipeline.create(dai.node.XLinkOut)
    xinSpatialConfig = pipeline.create(dai.node.XLinkIn)
    
    xoutRgb.setStreamName("rgb")
    xoutSpatialData.setStreamName("spatialData")
    xinSpatialConfig.setStreamName("spatialConfig")
    
    camRgb.setPreviewSize(640, 480)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camRgb.setFps(15) 
    
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoLeft.setFps(15)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    monoRight.setFps(15)
    
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    
    config = dai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = 100
    config.depthThresholds.upperThreshold = 10000
    config.roi = dai.Rect(dai.Point2f(0.49, 0.49), dai.Point2f(0.51, 0.51))
    spatialCalc.initialConfig.addROI(config)
    
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    camRgb.preview.link(xoutRgb.input)
    stereo.depth.link(spatialCalc.inputDepth)
    xinSpatialConfig.out.link(spatialCalc.inputConfig)
    spatialCalc.out.link(xoutSpatialData.input)
    
    # ---------------- 启动与推理循环 ----------------
    print("正在连接 OAK 相机... 将使用 USB 2.0 模式并恢复 3D 深度测距！")
    with dai.Device(pipeline, usb2Mode=True) as device:
        print(f"✅ 成功连接到 OAK 相机！USB 速度: {device.getUsbSpeed().name}")
        qRgb = device.getOutputQueue("rgb", 4, False)
        qSpatial = device.getOutputQueue("spatialData", 4, False)
        qConfig = device.getInputQueue("spatialConfig")
        
        print("🚀 纯 YOLO + 3D 空间坐标系统已启动！")
        
        while True:
            inRgb = qRgb.get()
            inSpatial = qSpatial.tryGet()
            frame = inRgb.getCvFrame()
            
            # --- 第一步：YOLO 精确定位 ---
            # 强化类过滤：classes=[0] 代表只识别 male
            results = model.predict(source=frame, show=False, imgsz=PREDICT_IMGSZ, 
                                    conf=PREDICT_CONF, iou=PREDICT_IOU, classes=[0], verbose=False)
            boxes = results[0].boxes
            
            best_conf = 0.0
            best_target = None
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    if conf > best_conf:
                        best_conf = conf
                        
                        # 直接抛弃二次校验，获取 YOLO 框的几何中心！
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0
                        best_target = (x1, y1, x2, y2, conf, cx, cy)
                        
            if best_target:
                x1, y1, x2, y2, conf, cx, cy = best_target
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, f"YOLO Male {conf:.2f}", (x1, max(30, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                # 通过 YOLO 中心设定深度获取 ROI 区域
                win_size_w = max(5.0, (x2 - x1) * 0.05)
                win_size_h = max(5.0, (y2 - y1) * 0.05)
                roi_x1 = max(0.0, (cx - win_size_w) / frame.shape[1])
                roi_y1 = max(0.0, (cy - win_size_h) / frame.shape[0])
                roi_x2 = min(1.0, (cx + win_size_w) / frame.shape[1])
                roi_y2 = min(1.0, (cy + win_size_h) / frame.shape[0])
                
                cfg = dai.SpatialLocationCalculatorConfigData()
                cfg.roi = dai.Rect(dai.Point2f(roi_x1, roi_y1), dai.Point2f(roi_x2, roi_y2))
                calcConfig = dai.SpatialLocationCalculatorConfig()
                calcConfig.addROI(cfg)
                qConfig.send(calcConfig)
                
            else:
                # 没识别到目标，重置探测 ROI
                calcConfig = dai.SpatialLocationCalculatorConfig()
                cfg = dai.SpatialLocationCalculatorConfigData()
                cfg.roi = dai.Rect(dai.Point2f(0.0, 0.0), dai.Point2f(0.01, 0.01))
                calcConfig.addROI(cfg)
                qConfig.send(calcConfig)
            
            if inSpatial is not None and best_target:
                x1, y1, x2, y2, conf, cx, cy = best_target
                data = inSpatial.getSpatialLocations()
                if data:
                    # 这里的 X, Y, Z 全是毫米级别的实际空间坐标，是以相机中心点为原点(0,0,0)测算的
                    # X 对应真实世界的左右，Y 对应真实世界的上下，Z 对应前后距离
                    X_mm = data[0].spatialCoordinates.x
                    Y_mm = data[0].spatialCoordinates.y
                    Z_mm = data[0].spatialCoordinates.z
                    
                    # 绘制十字准星标出画面中心
                    cv2.circle(frame, (320, 240), 2, (255, 0, 0), -1)
                    cv2.line(frame, (320-10, 240), (320+10, 240), (255, 0, 0), 1)
                    cv2.line(frame, (320, 240-10), (320, 240+10), (255, 0, 0), 1)
                    
                    cv2.line(frame, (320, 240), (int(cx), int(cy)), (0, 255, 255), 1)
                    cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
                    
                    # 红色数字显示物理偏移坐标，这就是距离相机中心的偏差，以“毫米(mm)”为单位！
                    cv2.putText(frame, f"Real Distance: X:{int(X_mm)} Y:{int(Y_mm)} Z:{int(Z_mm)} mm", 
                                (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)
                    
            cv2.imshow("Robot Arm Tracker (Real Millimeters)", frame)
            
            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()