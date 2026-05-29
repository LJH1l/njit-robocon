import cv2
import numpy as np
import depthai as dai

def find_grey_body(frame):
    """
    专门用来在复杂背景中寻找这个接头的灰色环部分。
    """
    # 1. 转换到 HSV 色彩空间，它对光线的明暗变化更有抵抗力
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 2. 定义“灰色”的 HSV 范围
    # 灰色在 HSV 中：饱和度 (S) 很低，亮度 (V) 属于中等（避开全白和全黑）
    # 如果实际光线有偏蓝或偏黄，你可以微调这几个值
    lower_grey = np.array([0, 0, 80])    # 最小饱和度, 最小亮度 (避开死黑)
    upper_grey = np.array([179, 60, 200])# 最大色调, 较低饱和度(避开彩色), 最大亮度(避开死白)
    
    # 根据灰色的范围生成一个遮罩 (只有灰色部分是白色的)
    mask = cv2.inRange(hsv, lower_grey, upper_grey)
    
    # 做一些形态学操作，去除画面中散落的小噪点，填补接头上的可能空洞
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 为了能清楚看到提取的情况，你可以稍后取消注释下面这行来调试颜色：
    # cv2.imshow("Debug Grey Mask", mask)
    
    # 3. 寻找这个白斑的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 过滤掉太小的灰点点
        if 800 < area < 30000:
            x, y, w, h = cv2.boundingRect(cnt)
            # 因为这个接头的灰环通常宽>高 或与高相近，可以做一定的比例过滤
            aspect_ratio = float(w) / h
            if 0.5 < aspect_ratio < 3.0: 
                valid_rects.append((x, y, w, h, cnt))
                
    return valid_rects

# --- DepthAI 管道设置 ---
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

# 相机参数配置
camRgb.setPreviewSize(640, 480)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# 深度对齐
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

# 初始计算器配置 (防止启动报空配置错)
config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 5000
config.roi = dai.Rect(dai.Point2f(0.5, 0.5), dai.Point2f(0.51, 0.51))
spatialCalc.initialConfig.addROI(config)

# 连线
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
camRgb.preview.link(xoutRgb.input)
stereo.depth.link(spatialCalc.inputDepth)
xinSpatialConfig.out.link(spatialCalc.inputConfig)
spatialCalc.out.link(xoutSpatialData.input)

# --- 运行设备 ---
print("正在连接相机...")
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue("rgb", 4, False)
    qSpatial = device.getOutputQueue("spatialData", 4, False)
    qConfig = device.getInputQueue("spatialConfig")

    print("程序已启动！尝试去识别灰色的连接头环。")

    while True:
        frame = qRgb.get().getCvFrame()
        inSpatial = qSpatial.tryGet()

        # 找灰色块
        valid_rects = find_grey_body(frame)
        
        newConfig = dai.SpatialLocationCalculatorConfig()
        found_roi = False
        
        if len(valid_rects) > 0:
            # 取面积最大的一块认为是接头主体
            valid_rects.sort(key=lambda r: r[2]*r[3], reverse=True)
            x, y, w, h, _ = valid_rects[0]
            
            # 画一个外框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, "Target: Grey Ring", (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 测距焦点设在框的最中间
            cx, cy = x + w/2, y + h/2
            roi_w, roi_h = w * 0.1, h * 0.1
            
            p1 = dai.Point2f(max(0.0, (cx - roi_w/2)/frame.shape[1]), max(0.0, (cy - roi_h/2)/frame.shape[0]))
            p2 = dai.Point2f(min(1.0, (cx + roi_w/2)/frame.shape[1]), min(1.0, (cy + roi_h/2)/frame.shape[0]))
            
            cf = dai.SpatialLocationCalculatorConfigData()
            cf.roi = dai.Rect(p1, p2)
            newConfig.addROI(cf)
            found_roi = True
            
            cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        else:
            cf = dai.SpatialLocationCalculatorConfigData()
            cf.roi = dai.Rect(dai.Point2f(0,0), dai.Point2f(0.01,0.01))
            newConfig.addROI(cf)
            
        qConfig.send(newConfig)

        # 现实坐标结果
        if inSpatial is not None and found_roi:
            data = inSpatial.getSpatialLocations()
            for d in data:
                X, Y, Z = d.spatialCoordinates.x, d.spatialCoordinates.y, d.spatialCoordinates.z
                cv2.putText(frame, f"X:{X:.0f} Y:{Y:.0f} Z:{Z:.0f}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Grey Connector Tracker", frame)
        if cv2.waitKey(1) == ord('q'):
            break
