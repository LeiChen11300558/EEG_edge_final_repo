# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import tflite_runtime.interpreter as tflite

NUM_THREADS = 4

"""
浠呮帹鐞嗙増鏈鏄庯細
- 璇诲彇 noisyinput_test.npy
- 浣跨敤 TFLite 妯″瀷閫愭潯鎺ㄧ悊
- 缁熻鎺ㄧ悊鏃堕棿
- 鍙€夛細灏嗘ā鍨嬭緭鍑轰繚瀛樹负 EEG_denoised.npy
"""

# ====================== 0. 鎺у埗鏄惁淇濆瓨杈撳嚭 ======================
SAVE_OUTPUT = False  # 鏀规垚 False 鍒欎笉浼氫繚瀛樻帹鐞嗙粨鏋?

# ====================== 1. 璺緞璁剧疆 ======================
current_path = os.path.dirname(os.path.abspath(__file__))

# 妯″瀷鏂囦欢鍚?
MODEL_NAME = "fu"   # 鑻ヤ负 fu.tflite锛岃鍐?"fu.tflite"

# data 鐩綍锛?./data
data_dir = os.path.join(current_path, "..", "data")

NOISY_NAME = "noiseinput_test.npy"
OUT_NAME = "EEG_denoised.npy"

model_path = os.path.join(current_path, MODEL_NAME)
noisy_path = os.path.join(data_dir, NOISY_NAME)
out_path   = os.path.join(data_dir, OUT_NAME)

print("Model path:", model_path)
print("Noisy  data path:", noisy_path)
print("Output data path:", out_path)

# ====================== 2. 璇诲彇 noisy 鏁版嵁 ======================
noisy = np.load(noisy_path)
print("Noisy shape:", noisy.shape, ", dtype:", noisy.dtype)

def to_2d(arr):
    """缁熶竴杞崲涓?(N,L) float32"""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    return arr

noisy_2d = to_2d(noisy)
N, segment_len = noisy_2d.shape
print("Using N = {}, segment_len = {}".format(N, segment_len))

# ====================== 3. 鍔犺浇 TFLite 妯″瀷 ======================
print(f"Creating TFLite interpreter with num_threads = {NUM_THREADS}")
interpreter = tflite.Interpreter(model_path=model_path, num_threads=NUM_THREADS)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite input_details:", input_details)
print("TFLite output_details:", output_details)

input_shape = input_details[0]['shape']

# ====================== 4. 浠呮帹鐞?+ 鏃堕棿缁熻 ======================
total_inference_time = 0.0
single_segment_inference_times = []
results = []

print("\n==== Start inference only ====")

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

# ====================== 5. 杈撳嚭鏁寸悊 ======================
results = np.array(results, dtype=np.float32)
decoded_2d = results.reshape(N, segment_len)

print("decoded_2d shape:", decoded_2d.shape, ", dtype:", decoded_2d.dtype)

# ---------------------- 淇濆瓨寮€鍏?----------------------
if SAVE_OUTPUT:
    np.save(out_path, decoded_2d)
    print("Saved denoised data to:", out_path)
else:
    print("SAVE_OUTPUT = False")

# ====================== 6. 鎺ㄧ悊鏃堕棿缁熻 ======================
single_segment_inference_times = np.array(single_segment_inference_times, dtype=np.float64)
avg_single = single_segment_inference_times.mean()
min_single = single_segment_inference_times.min()
max_single = single_segment_inference_times.max()

print("\n==== Inference time ====")
print("Total inference time: {:.6f} s".format(total_inference_time))
print("Average inference time per segment: {:.6f} s ({:.2f} ms)".format(
    avg_single, avg_single * 1000.0))
print("Min / Max per-segment time: {:.6f} s / {:.6f} s".format(
    min_single, max_single))
print("First segment inference time: {:.6f} s ({:.2f} ms)".format(
    single_segment_inference_times[0], single_segment_inference_times[0] * 1000.0))
print("Inference only finished.")
