import socket
import msgpack
import time
import numpy as np
import struct
import tflite_runtime.interpreter as tflite
import os

class BridgeClient:
    """直接通过 Unix socket 与 arduino-router 通信"""
    def __init__(self, addr="/var/run/arduino-router.sock"):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(addr)
        self.unpacker = msgpack.Unpacker(raw=False)
        self.msg_id = 0
    
    def call(self, method, *args, timeout=10):
        self.msg_id += 1
        # MsgPack-RPC: [type=0, msgid, method, params]
        msg = msgpack.packb([0, self.msg_id, method, list(args)])
        self.sock.sendall(msg)
        
        # 读取响应
        self.sock.settimeout(timeout)
        while True:
            data = self.sock.recv(4096)
            if not data:
                raise ConnectionError("Socket closed")
            self.unpacker.feed(data)
            for resp in self.unpacker:
                # [type=1, msgid, error, result]
                if resp[0] == 1 and resp[1] == self.msg_id:
                    if resp[2] is not None:
                        raise RuntimeError(f"RPC error: {resp[2]}")
                    return resp[3]
    
    def close(self):
        self.sock.close()

# === 加载模型和数据 ===
current_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_path, "daefloat")
noisy_path = os.path.join(current_path, "x_test_noisy1.npy")
clean_path = os.path.join(current_path, "x_test_clean1.npy")

print("Loading model and data...")
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

x_noisy = np.load(noisy_path).astype(np.float32)
x_clean = np.load(clean_path).astype(np.float32)
print(f"Loaded {x_noisy.shape[0]} samples, input shape: {input_details[0]['shape']}")

# === 连接 Bridge ===
print("Connecting to Bridge router...")
bridge = BridgeClient()
print("Connected!")

# === Pipeline 测试 ===
NUM_SAMPLES = 50

bridge_times = []
inference_times = []
total_times = []

print(f"\n=== Pipeline Test: {NUM_SAMPLES} samples ===")
print("Flow: call MCU echo -> TFLite inference\n")

for i in range(NUM_SAMPLES):
    t_total_start = time.perf_counter()

    # Bridge 往返
    t_bridge_start = time.perf_counter()
    result = bridge.call("echo_index", i)
    t_bridge_end = time.perf_counter()

    # TFLite 推理
    t_infer_start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], x_noisy[i].reshape(1, 800))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    t_infer_end = time.perf_counter()

    t_total_end = time.perf_counter()

    bridge_ms = (t_bridge_end - t_bridge_start) * 1000
    infer_ms = (t_infer_end - t_infer_start) * 1000
    total_ms = (t_total_end - t_total_start) * 1000

    bridge_times.append(bridge_ms)
    inference_times.append(infer_ms)
    total_times.append(total_ms)

    if (i + 1) % 10 == 0:
        print(f"  [{i+1}/{NUM_SAMPLES}] bridge={bridge_ms:.1f}ms infer={infer_ms:.1f}ms total={total_ms:.1f}ms")

bridge.close()

bridge_times = np.array(bridge_times)
inference_times = np.array(inference_times)
total_times = np.array(total_times)

print(f"\n{'='*50}")
print(f"  Results ({NUM_SAMPLES} samples)")
print(f"{'='*50}")
print(f"  Bridge roundtrip:  avg={bridge_times.mean():.2f}ms  min={bridge_times.min():.2f}ms  max={bridge_times.max():.2f}ms")
print(f"  TFLite inference:  avg={inference_times.mean():.2f}ms  min={inference_times.min():.2f}ms  max={inference_times.max():.2f}ms")
print(f"  Total per sample:  avg={total_times.mean():.2f}ms  min={total_times.min():.2f}ms  max={total_times.max():.2f}ms")
print(f"  Max throughput:    {1000/total_times.mean():.1f} samples/sec")
print(f"{'='*50}")
