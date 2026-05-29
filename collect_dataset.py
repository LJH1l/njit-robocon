import cv2
import depthai as dai
import os
import time

# 创建保存图片的文件夹
save_dir = "dataset/images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 创建管道
pipeline = dai.Pipeline()

# 配置彩色相机
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# 设置预览界面的大小（最好是 YOLO 所需的方形比例或接近的比例）
camRgb.setPreviewSize(640, 640)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# 默认开启连续自动对焦
camRgb.initialControl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)

# 控制流（用于手动触发对焦等操作）
controlIn = pipeline.create(dai.node.XLinkIn)
controlIn.setStreamName('control')
controlIn.out.link(camRgb.inputControl)

# 输出流
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

# 启动设备
print("正在启动相机...")
with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qControl = device.getInputQueue(name="control", maxSize=1, blocking=False)
    
    img_count = 0
    print(f"数据采集程序已就绪！")
    print(f"👉 按键盘 's' 键保存当前画面。")
    print(f"👉 按键盘 'f' 键手动触发一次自动对焦。")
    print(f"👉 按键盘 'q' 键退出程序。")
    print(f"请变换接头的位置、翻转它们，或者加入几张两个接头都在画面里的照片。")

    while True:
        inPreview = qRgb.get()
        frame = inPreview.getCvFrame()

        # 显示画面
        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('f'):
            # 触发一次自动对焦
            ctrl = dai.CameraControl()
            ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
            ctrl.setAutoFocusTrigger()
            qControl.send(ctrl)
            print("[操作] 已重新触发自动对焦...")
        elif key == ord('s'):
            # 生成带时间戳的文件名防止覆盖
            filename = f"img_{int(time.time())}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, frame)
            img_count += 1
            print(f"[成功] 保存了第 {img_count} 张照片: {filepath}")

print(f"采集结束，共采集了 {img_count} 张照片。")
