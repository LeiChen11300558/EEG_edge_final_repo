# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import tflite_runtime.interpreter as tflite

NUM_THREADS = 4

"""
仅推理版本说明：
- 读取 noisyinput_test.npy
- 使用 TFLite 模型逐条推理
- 统计推理时间
- 可选：将模型输出保存为 EEG_denoised.npy
"""

# ====================== 0. 控制是否保存输出 ======================
SAVE_OUTPUT = False  # 改成 False 则不会保存推理结果

# ====================== 1. 路径设置 ======================
current_path = os.path.dirname(os.path.abspath(__file__))

# 模型文件名
MODEL_NAME = "fu"   # 若为 fu.tflite，请写 "fu.tflite"

# data 目录：../data
data_dir = os.path.join(current_path, "..", "data")

NOISY_NAME = "noiseinput_test.npy"
OUT_NAME = "EEG_denoised.npy"

model_path = os.path.join(current_path, MODEL_NAME)
noisy_path = os.path.join(data_dir, NOISY_NAME)
out_path   = os.path.join(data_dir, OUT_NAME)

print("Model path:", model_path)
print("Noisy  data path:", noisy_path)
print("Output data path:", out_path)

# ====================== 2. 读取 noisy 数据 ======================
noisy = np.load(noisy_path)
print("Noisy shape:", noisy.shape, ", dtype:", noisy.dtype)

def to_2d(arr):
    """统一转换为 (N,L) float32"""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    return arr

noisy_2d = to_2d(noisy)
N, segment_len = noisy_2d.shape
print("Using N = {}, segment_len = {}".format(N, segment_len))

# ====================== 3. 加载 TFLite 模型 ======================
print(f"Creating TFLite interpreter with num_threads = {NUM_THREADS}")
interpreter = tflite.Interpreter(model_path=model_path, num_threads=NUM_THREADS)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite input_details:", input_details)
print("TFLite output_details:", output_details)

input_shape = input_details[0]['shape']

# ====================== 4. 仅推理 + 时间统计 ======================
total_inference_time = 0.0
single_segment_inference_times = []
results = []

print("\n==== Start inference only ====")
start_all = time.time()

for idx, segment in enumerate(noisy_2d):
    segment = segment.astype(np.float32)

    if len(input_shape) == 3:
        segment_reshaped = segment.reshape(1, segment_len, 1)
    elif len(input_shape) == 2:
        segment_reshaped = segment.reshape(1, segment_len)
    else:
        raise ValueError("Unsupported input_shape: {}".format(input_shape))

    interpreter.set_tensor(input_details[0]['index'], segment_reshaped)

    t0 = time.time()
    interpreter.invoke()
    t1 = time.time()

    dt = t1 - t0
    total_inference_time += dt
    single_segment_inference_times.append(dt)

    out = interpreter.get_tensor(output_details[0]['index'])
    out = np.squeeze(out)
    results.append(out)

end_all = time.time()
wall_clock_time = end_all - start_all

# ====================== 5. 输出整理 ======================
results = np.array(results, dtype=np.float32)
decoded_2d = results.reshape(N, segment_len)

print("decoded_2d shape:", decoded_2d.shape, ", dtype:", decoded_2d.dtype)

# ---------------------- 保存开关 ----------------------
if SAVE_OUTPUT:
    np.save(out_path, decoded_2d)
    print("Saved denoised data to:", out_path)
else:
    print("SAVE_OUTPUT = False")

# ====================== 6. 推理时间统计 ======================
single_segment_inference_times = np.array(single_segment_inference_times, dtype=np.float64)
avg_single = single_segment_inference_times.mean()
min_single = single_segment_inference_times.min()
max_single = single_segment_inference_times.max()

print("\n==== Inference time (invoke only) ====")
print("Total invoke time: {:.6f} s".format(total_inference_time))
print("Average per segment: {:.6f} s ({:.2f} ms)".format(
    avg_single, avg_single * 1000.0))
print("Min / Max per segment: {:.6f} s / {:.6f} s".format(
    min_single, max_single))
print("First segment: {:.6f} s ({:.2f} ms)".format(
    single_segment_inference_times[0], single_segment_inference_times[0] * 1000.0))

print("\nWall-clock time (including loop overhead): {:.6f} s".format(wall_clock_time))
print("Inference only finished.")
