import depthai as dai

print("正在扫描可用的 OAK 相机设备...")
devices = dai.Device.getAllAvailableDevices()

if len(devices) == 0:
    print("❌ 错误：电脑没有扫描到任何 OAK 设备！")
    print("可能的原因：")
    print("1. 数据线不是具有数据传输功能的 USB 3.0 Type-C 线。")
    print("2. 接口供电不足，或者插在了不支持扩展的拓展坞上。")
    print("3. 相机没插紧。")
else:
    print(f"✅ 成功扫描到 {len(devices)} 个设备：")
    for device in devices:
        print(f" - 设备 ID: {device.getMxId()}, 状态: {device.state}")
