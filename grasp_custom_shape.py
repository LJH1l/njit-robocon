import cv2
import numpy as np
import depthai as dai

def find_target_contours(frame, config_color='any'):
    """
    使用 OpenCV 在画面中寻找可能的零件轮廓。
    针对灰白色桌面或红/蓝色比赛场地。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if config_color == 'any':
        # 基础灰度阈值：假设零件通常比白灰色桌面暗
        # 你可能需要根据实际灯光调整这里的 100
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    elif config_color == 'red':
        # 在实际比赛中，如果是红色背景找黑色/灰色零件，需要用不同的 HSV 阈值
        pass
    elif config_color == 'blue':
        pass

    # 寻找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_rects = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 过滤掉太小或太大的噪点 (面积需要根据相机距离调整)
        if 500 < area < 20000:
            # 找到外接矩形
            x, y, w, h = cv2.boundingRect(cnt)
            # 简单过滤长宽比，防止识别到奇形怪状的影子
            aspect_ratio = float(w) / h
            if 0.2 < aspect_ratio < 5.0:
                 valid_rects.append((x, y, w, h, cnt))
                 
    return valid_rects

# 1. 创建 DepthAI 管道
pipeline = dai.Pipeline()

# 2. 定义节点
camRgb = pipeline.create(dai.node.ColorCamera)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
spatialLocationCalculator = pipeline.create(dai.node.SpatialLocationCalculator)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutSpatialData = pipeline.create(dai.node.XLinkOut)
xinSpatialCalcConfig = pipeline.create(dai.node.XLinkIn)

xoutRgb.setStreamName("rgb")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# 3. 配置相机参数
camRgb.setPreviewSize(640, 480) # 方便OpenCV处理的适中分辨率
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# 深度配置
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A) # 深度图必须和RGB对齐！

# 配置空间位置计算器 (SpatialLocationCalculator) 默认参数
# 我们会在运行时动态更新它的测量区域 (ROI)
config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 5000
# 初始化一个全屏的 ROI（随便设的，运行中会被覆盖）
config.roi = dai.Rect(dai.Point2f(0.4, 0.4), dai.Point2f(0.6, 0.6))
spatialLocationCalculator.initialConfig.addROI(config)

# 4. 连接节点路由图
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# RGB 直出到电脑，OpenCV来找轮廓
camRgb.preview.link(xoutRgb.input)

# 将深度图输送给计算器节点
stereo.depth.link(spatialLocationCalculator.inputDepth)

# 允许从电脑动态传入 ROI 区域配置
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)
# 计算器输出结果
spatialLocationCalculator.out.link(xoutSpatialData.input)

# 5. 运行设备
print("正在连接相机...")
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qSpatialData = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    qSpatialConfig = device.getInputQueue("spatialCalcConfig")

    print("程序已启动！按 'q' 退出。")

    while True:
        inRgb = qRgb.get()
        inSpatialData = qSpatialData.tryGet()

        frame = inRgb.getCvFrame()
        
        # --- 第1步：用 OpenCV 寻找零件大致位置 ---
        # 寻找轮廓
        valid_rects = find_target_contours(frame)
        
        # --- 第2步：将找到的位置框发送给相机，让它测量深度 ---
        newConfig = dai.SpatialLocationCalculatorConfig()
        
        found_roi = False
        if len(valid_rects) > 0:
            # 取面积最大的那个轮廓作为主要目标
            valid_rects.sort(key=lambda r: w*h if (w:=r[2]) and (h:=r[3]) else 0, reverse=True)
            x, y, w, h, _ = valid_rects[0]
            
            # 画绿色的 2D 框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 缩小测量范围到中心区域，避免测到边缘的背景深度
            center_x = x + w/2
            center_y = y + h/2
            roi_w = w * 0.2
            roi_h = h * 0.2
            
            roi_left = max(0, int(center_x - roi_w/2))
            roi_top = max(0, int(center_y - roi_h/2))
            roi_right = min(frame.shape[1]-1, int(center_x + roi_w/2))
            roi_bottom = min(frame.shape[0]-1, int(center_y + roi_h/2))

            # 画个红点表示测距的心
            cv2.circle(frame, (int(center_x), int(center_y)), 3, (0, 0, 255), -1)

            # 将像素坐标归一化到 0.0 ~ 1.0 (相机的要求)
            conf_data = dai.SpatialLocationCalculatorConfigData()
            conf_data.depthThresholds.lowerThreshold = 100
            conf_data.depthThresholds.upperThreshold = 5000
            conf_data.roi = dai.Rect(dai.Point2f(roi_left / frame.shape[1], roi_top / frame.shape[0]),
                                     dai.Point2f(roi_right / frame.shape[1], roi_bottom / frame.shape[0]))
            newConfig.addROI(conf_data)
            found_roi = True
        else:
            # 如果没找到，给个默认的不显示的 ROI 避免报错
            conf_data = dai.SpatialLocationCalculatorConfigData()
            conf_data.roi = dai.Rect(dai.Point2f(0, 0), dai.Point2f(0.01, 0.01))
            newConfig.addROI(conf_data)

        # 把新的 ROI 发送给相机
        qSpatialConfig.send(newConfig)

        # --- 第3步：读取相机计算返回的 3D 坐标 ---
        if inSpatialData is not None and found_roi:
            spatialData = inSpatialData.getSpatialLocations()
            for depthData in spatialData:
                # 只有当我们自己发送的 ROI 有计算结果时才显示
                z_mm = depthData.spatialCoordinates.z
                x_mm = depthData.spatialCoordinates.x
                y_mm = depthData.spatialCoordinates.y
                
                # 在画面上标出 3D 坐标
                cv2.putText(frame, f"X:{x_mm:.0f}mm", (x, max(20, y - 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Y:{y_mm:.0f}mm", (x, max(20, y - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Z:{z_mm:.0f}mm", (x, max(20, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Shape Based 3D Grasping", frame)

        if cv2.waitKey(1) == ord('q'):
            break
