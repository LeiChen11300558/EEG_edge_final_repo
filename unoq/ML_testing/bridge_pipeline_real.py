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
                # 跳过 notification (type=2) 只处理 response (type=1)
                if not isinstance(resp, list):
                    continue
                if resp[0] == 1 and resp[1] == self.msg_id:
                    if resp[2] is not None:
                        raise RuntimeError(f"err: {resp[2]}")
                    return resp[3]
                # 跳过其他消息继续等

    def close(self):
        self.sock.close()

current_path = os.path.dirname(os.path.abspath(__file__))
print("Loading model and data...")
interpreter = tflite.Interpreter(model_path=os.path.join(current_path, "daefloat"))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

x_noisy = np.load(os.path.join(current_path, "x_test_noisy1.npy")).astype(np.float32)
print(f"Loaded {x_noisy.shape[0]} samples")

POINTS_PER_CALL = 8
CALLS_PER_SAMPLE = 100  # 800 / 8
NUM_SAMPLES = 5

bridge = BridgeClient()
print("Connected!\n")

# 先验证单次 call
test_ret = bridge.call("echo_data", "0.12,0.34,0.56")
print(f"Verify: sent '0.12,0.34,0.56' got '{test_ret}' type={type(test_ret)}\n")

print(f"=== Real EEG Pipeline: {NUM_SAMPLES} samples ===\n")

bridge_times = []
inference_times = []
total_times = []

for i in range(NUM_SAMPLES):
    t_total_start = time.perf_counter()
    received = np.zeros(800, dtype=np.float32)

    t_bridge_start = time.perf_counter()
    for c in range(CALLS_PER_SAMPLE):
        start = c * POINTS_PER_CALL
        end = start + POINTS_PER_CALL
        chunk = ",".join(f"{v:.2f}" for v in x_noisy[i][start:end])
        ret = bridge.call("echo_data", chunk)
        received[start:end] = [float(x) for x in ret.split(",")]
    t_bridge_end = time.perf_counter()

    match = np.allclose(x_noisy[i], received, atol=0.01)

    t_infer_start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], received.reshape(1, 800))
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
    print(f"  [{i+1}/{NUM_SAMPLES}] bridge={bridge_ms:.0f}ms infer={infer_ms:.1f}ms total={total_ms:.0f}ms match={match}")

bridge.close()

bridge_times = np.array(bridge_times)
inference_times = np.array(inference_times)
total_times = np.array(total_times)

print(f"\n{'='*55}")
print(f"  Results ({NUM_SAMPLES} samples, real npy data)")
print(f"{'='*55}")
print(f"  Bridge (100x8pts):   avg={bridge_times.mean():.0f}ms")
print(f"  TFLite inference:    avg={inference_times.mean():.1f}ms")
print(f"  Total per sample:    avg={total_times.mean():.0f}ms")
print(f"{'='*55}")
