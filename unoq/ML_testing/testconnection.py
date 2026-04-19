import gpiod
from gpiod.line import Direction, Value
import time
import sys

# === 你的 JMEDIA 引脚定义 ===
# 注意：通常高通 SOC 的主要 GPIO 都在 gpiochip1 或 gpiochip0 上。
# 如果报错“请求引脚失败”，请把这里的 1 改成 0 试试！
CHIP_PATH = '/dev/gpiochip1' 

SCLK_PIN = 29
MOSI_PIN = 22
MISO_PIN = 23
CS_PIN   = 30
# DRDY_PIN = 20 # 读 ID 暂时用不上 DRDY，为了防止占用报错，我们先注释掉

def test_ads1299():
    print(f"尝试连接 {CHIP_PATH} 并配置引脚...")
    try:
        # 使用 gpiod v2 的新语法请求引脚
        req = gpiod.request_lines(
            CHIP_PATH,
            consumer="ads1299-test",
            config={
                SCLK_PIN: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE), # SCLK 初始拉低
                MOSI_PIN: gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.INACTIVE), # MOSI 初始拉低
                CS_PIN:   gpiod.LineSettings(direction=Direction.OUTPUT, output_value=Value.ACTIVE),   # CS 初始拉高 (不选中)
                MISO_PIN: gpiod.LineSettings(direction=Direction.INPUT)                                # MISO 设为输入
            }
        )
    except Exception as e:
        print(f"❌ 引脚配置失败: {e}")
        print("💡 提示：可能是 gpiochip 编号不对（试试改成 /dev/gpiochip0），或者引脚被其他程序占用了。")
        sys.exit(1)

    print("✅ 引脚配置成功！开始发送 SPI 指令...")

    def spi_transfer(byte_out):
        """纯软件模拟 SPI 通信 (Mode 1)"""
        byte_in = 0
        for i in range(7, -1, -1):
            # 1. 准备发出的数据位 (MOSI)
            bit_out = Value.ACTIVE if (byte_out & (1 << i)) else Value.INACTIVE
            req.set_value(MOSI_PIN, bit_out)
            
            # 2. 时钟上升沿
            req.set_value(SCLK_PIN, Value.ACTIVE)
            
            # 3. 时钟下降沿，同时读取 MISO 上的数据
            req.set_value(SCLK_PIN, Value.INACTIVE)
            if req.get_value(MISO_PIN) == Value.ACTIVE:
                byte_in |= (1 << i)
                
        return byte_in

    # --- 开始与 ADS1299 对话 ---
    try:
        # 1. 拉低 CS，叫醒芯片
        req.set_value(CS_PIN, Value.INACTIVE)
        
        # 2. 发送 SDATAC (0x11) 命令，停止连续读取，允许读写寄存器
        spi_transfer(0x11)
        
        # 3. 拉高 CS，给芯片一点反应时间
        req.set_value(CS_PIN, Value.ACTIVE)
        time.sleep(0.1)
        
        # 4. 再次拉低 CS，准备读 ID
        req.set_value(CS_PIN, Value.INACTIVE)
        
        # 读寄存器命令: 0x20 (RREG) | 0x00 (寄存器地址)
        spi_transfer(0x20 | 0x00) 
        # 告诉芯片我们要读 1 个字节 (发送 0x00 表示读 1 字节)
        spi_transfer(0x00)        
        # 发送空数据 (0x00) 把芯片里的 ID 挤出来
        device_id = spi_transfer(0x00) 
        
        # 5. 通信结束，拉高 CS
        req.set_value(CS_PIN, Value.ACTIVE)

        # --- 结果宣判 ---
        print("-" * 40)
        print(f"📡 读到的芯片 ID 是: {hex(device_id)}")
        
        if device_id == 0x3E:
            print("🎉 太牛了！成功读到 0x3E，完美识别到 ADS1299 (8通道版)！")
            print("🎉 你的硬件连线（包括飞线）、电平转换和软件链路已经全部打通！")
        elif device_id in [0x3C, 0x3D]:
             print("🎉 完美！成功识别到 ADS1299 (少通道版)！")
        elif device_id == 0x00 or device_id == 0xff:
            print("❌ 失败：读出全是 0 或全是 1。")
            print("排查重点：1. 飞线的 1.8V 供电是否正常？ 2. SN74 的方向(DIR)是不是对的？ 3. RESET 是不是没拉高？")
        else:
            print("⚠️ 读到了奇怪的数据，可能是线虚接或者电磁干扰。")
        print("-" * 40)

    finally:
        # 释放引脚，防止下次运行报错
        req.release()

if __name__ == "__main__":
    test_ads1299()