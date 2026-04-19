import socket, msgpack, time
import numpy as np
import tflite_runtime.interpreter as tflite
import os

class BridgeClient:
    def __init__(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect("/var/run/arduino-router.sock")
        self.unpacker = msgpack.Unpacker(raw=False)
        self.msg_id = 0

    def call(self, method, *args, timeout=5):
        self.msg_id += 1
        msg = msgpack.packb([0, self.msg_id, method, list(args)])
        self.sock.sendall(msg)
        self.sock.settimeout(timeout)
        while True:
            data = self.sock.recv(65536)
            if not data:
                raise ConnectionError("closed")
            self.unpacker.feed(data)
            for resp in self.unpacker:
                if not isinstance(resp, list):
                    continue
                if resp[0] == 1 and resp[1] == self.msg_id:
                    if resp[2] is not None:
                        raise RuntimeError(f"err: {resp[2]}")
                    return resp[3]

    def close(self):
        self.sock.close()

current_path = os.path.dirname(os.path.abspath(__file__))
interpreter = tflite.Interpreter(model_path=os.path.join(current_path, "daefloat"))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

x_noisy = np.load(os.path.join(current_path, "x_test_noisy1.npy")).astype(np.float32)
x_clean = np.load(os.path.join(current_path, "x_test_clean1.npy")).astype(np.float32)

bridge = BridgeClient()

# 传输 1 个样本的真实数据
print("Sending sample 0 through MCU...")
received = np.zeros(800, dtype=np.float32)
for c in range(100):
    start = c * 8
    end = start + 8
    chunk = ",".join(f"{v:.4f}" for v in x_noisy[0][start:end])
    ret = bridge.call("echo_data", chunk)
    received[start:end] = [float(x) for x in ret.split(",")]
bridge.close()

# 用回传数据推理
interpreter.set_tensor(input_details[0]['index'], received.reshape(1, 800))
interpreter.invoke()
output_bridge = interpreter.get_tensor(output_details[0]['index']).flatten()

# 直接用本地数据推理（对照）
interpreter.set_tensor(input_details[0]['index'], x_noisy[0].reshape(1, 800))
interpreter.invoke()
output_local = interpreter.get_tensor(output_details[0]['index']).flatten()

print(f"\n{'='*60}")
print(f"  Sample 0 Results")
print(f"{'='*60}")
print(f"\n  Noisy input (first 10):  {x_noisy[0][:10]}")
print(f"  Clean target (first 10): {x_clean[0][:10]}")
print(f"  Bridge output (first 10):{output_bridge[:10]}")
print(f"  Local output (first 10): {output_local[:10]}")

# 比较
bridge_vs_local = np.allclose(output_bridge, output_local, atol=1e-4)
print(f"\n  Bridge vs Local match: {bridge_vs_local}")

# 去噪质量
def rrmse(true, pred):
    return np.sqrt(np.mean((true - pred)**2)) / (np.sqrt(np.mean(true**2)) + 1e-10)

def cc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return np.sum(a * b) / (np.sqrt(np.sum(a**2) * np.sum(b**2)) + 1e-10)

# 零中心化
noisy_z = x_noisy[0] - x_noisy[0].mean()
clean_z = x_clean[0] - x_clean[0].mean()
output_z = output_bridge - output_bridge.mean()

print(f"\n  --- Denoising Quality (Sample 0) ---")
print(f"  Noisy vs Clean:   RRMSE={rrmse(clean_z, noisy_z):.4f}  CC={cc(clean_z, noisy_z):.4f}")
print(f"  Denoised vs Clean: RRMSE={rrmse(clean_z, output_z):.4f}  CC={cc(clean_z, output_z):.4f}")

if rrmse(clean_z, output_z) < rrmse(clean_z, noisy_z):
    print(f"  ✅ Denoising improved RRMSE")
else:
    print(f"  ⚠️ Denoising did not improve RRMSE (sample may be clean)")

print(f"{'='*60}")
