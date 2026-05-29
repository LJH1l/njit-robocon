import cv2
import numpy as np
import depthai as dai

def empty_callback(val):
    pass

# 在最开始就创建好调节滑块的窗口
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 500, 300)
# HSV 默认的灰度提取参数
cv2.createTrackbar("H_MIN", "Trackbars", 0, 179, empty_callback)
cv2.createTrackbar("H_MAX", "Trackbars", 179, 179, empty_callback)
cv2.createTrackbar("S_MIN", "Trackbars", 0, 255, empty_callback)
cv2.createTrackbar("S_MAX", "Trackbars", 60, 255, empty_callback)    # 照片测算：饱和度在20~40左右
cv2.createTrackbar("V_MIN", "Trackbars", 80, 255, empty_callback)    # 照片测算：灰色亮度远高于纯黑(0-50)
cv2.createTrackbar("V_MAX", "Trackbars", 170, 255, empty_callback)   # 照片测算：灰色亮度在100~160之间，低于白色背景(200+)

def find_grey_body(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 实时读取滑块的值
    try:
        h_min = cv2.getTrackbarPos("H_MIN", "Trackbars")
        s_min = cv2.getTrackbarPos("S_MIN", "Trackbars")
        v_min = cv2.getTrackbarPos("V_MIN", "Trackbars")
        h_max = cv2.getTrackbarPos("H_MAX", "Trackbars")
        s_max = cv2.getTrackbarPos("S_MAX", "Trackbars")
        v_max = cv2.getTrackbarPos("V_MAX", "Trackbars")
    except:
        h_min, s_min, v_min = 0, 0, 80
        h_max, s_max, v_max = 179, 60, 200

    lower_grey = np.array([h_min, s_min, v_min])
    upper_grey = np.array([h_max, s_max, v_max])
    
    # 获取遮罩 (Mask)
    mask = cv2.inRange(hsv, lower_grey, upper_grey)
    
    # 形态学去噪
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 弹出一个 Debug 窗口，专门显示提取出来的像素！白色的代表识别到的区域
    cv2.imshow("HSV Mask Debug", mask)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 400 < area < 40000: # 放宽一点面积限制方便调参
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 0.3 < aspect_ratio < 4.0: 
                valid_rects.append((x, y, w, h, cnt))
                
    return valid_rects


# --- DepthAI 管道基础设置 ---
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

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 5000
config.roi = dai.Rect(dai.Point2f(0.5, 0.5), dai.Point2f(0.51, 0.51))
spatialCalc.initialConfig.addROI(config)

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

    print("程序已启动！")
    print("请看着【HSV Mask Debug】窗口：如果里面接头的灰色段变成了纯白的一片，而背景全是黑的，就说明参数对了！")

    while True:
        frame = qRgb.get().getCvFrame()
        inSpatial = qSpatial.tryGet()

        valid_rects = find_grey_body(frame)
        
        newConfig = dai.SpatialLocationCalculatorConfig()
        found_roi = False
        
        if len(valid_rects) > 0:
            valid_rects.sort(key=lambda r: r[2]*r[3], reverse=True)
            x, y, w, h, _ = valid_rects[0]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, "Target", (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

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

        if inSpatial is not None and found_roi:
            data = inSpatial.getSpatialLocations()
            for d in data:
                X, Y, Z = d.spatialCoordinates.x, d.spatialCoordinates.y, d.spatialCoordinates.z
                cv2.putText(frame, f"X:{X:.0f} Y:{Y:.0f} Z:{Z:.0f}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Grey Connector Tracker", frame)
        if cv2.waitKey(1) == ord('q'):
            break
