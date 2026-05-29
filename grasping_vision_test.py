import cv2
import depthai as dai
import blobconverter

# 1. 创建 DepthAI 管道
pipeline = dai.Pipeline()

# 2. 定义节点：彩色相机、双目深度、AI目标检测网络
camRgb = pipeline.create(dai.node.ColorCamera)
spatialDetectionNetwork = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")

# 3. 配置彩色相机参数 (修改了旧版的常量以消除 DeprecationWarning)
camRgb.setPreviewSize(300, 300)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

# 配置双目相机参数
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# 设定双目深度，并将深度图对齐到彩色 RGB 镜头
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())

# AI配置：自动下载一个轻量级的目标检测网络(MobileNet SSD)
spatialDetectionNetwork.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

# 4. 连接节点路由图
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
spatialDetectionNetwork.passthrough.link(xoutRgb.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.out.link(xoutNN.input)

# 5. 连接设备并运行
print("正在连接相机并加载AI模型，请稍候...")
with dai.Device(pipeline) as device:
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)

    print("程序已启动！按键盘 'q' 键退出。")
    print("请将水杯(cup)、瓶子(bottle)或手机等物体放在摄像头前...")

    while True:
        inPreview = previewQueue.get()
        inDet = detectionNNQueue.get()

        frame = inPreview.getCvFrame()
        detections = inDet.detections

        for det in detections:
            x1 = int(det.xmin * frame.shape[1])
            y1 = int(det.ymin * frame.shape[0])
            x2 = int(det.xmax * frame.shape[1])
            y2 = int(det.ymax * frame.shape[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            x_mm = det.spatialCoordinates.x
            y_mm = det.spatialCoordinates.y
            z_mm = det.spatialCoordinates.z

            cv2.putText(frame, f"X: {x_mm:.0f} mm", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Y: {y_mm:.0f} mm", (x1, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Z: {z_mm:.0f} mm", (x1, y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Vision Guided Grasping - Debug", frame)

        if cv2.waitKey(1) == ord('q'):
            break
