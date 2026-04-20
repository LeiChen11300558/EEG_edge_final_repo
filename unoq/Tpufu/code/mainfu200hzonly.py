# -*- coding: utf-8 -*-
import numpy as np
import os
import time
import tflite_runtime.interpreter as tflite


def resample_1d_batch(data, target_len):
    data = np.asarray(data, dtype=np.float32)
    orig_len = data.shape[1]
    if orig_len == target_len:
        return data

    old_x = np.linspace(0.0, 1.0, orig_len, dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    out = np.empty((data.shape[0], target_len), dtype=np.float32)
    for i in range(data.shape[0]):
        out[i] = np.interp(new_x, old_x, data[i]).astype(np.float32)
    return out

# ====================== 1. 璺緞涓庢ā鍨嬭缃?======================
SAVE_OUTPUT = False  
current_path = os.path.dirname(os.path.abspath(__file__))

# ---> 淇敼涓轰綘鏂颁笂浼犵殑妯″瀷鍚嶇О <---
MODEL_NAME = "fu200hz"   

data_dir = os.path.join(current_path, "..", "data")
NOISY_NAME = "noiseinput_test.npy"
OUT_NAME = "EEG_denoised_200hz.npy"

model_path = os.path.join(current_path, MODEL_NAME)
noisy_path = os.path.join(data_dir, NOISY_NAME)
out_path   = os.path.join(data_dir, OUT_NAME)

print("Model path:", model_path)
print("Noisy data path:", noisy_path)

# ====================== 2. 鍔犺浇 TFLite 妯″瀷 ======================
NUM_THREADS = 4
print(f"Creating TFLite interpreter with num_threads = {NUM_THREADS}")
interpreter = tflite.Interpreter(model_path=model_path, num_threads=NUM_THREADS)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 鑷姩鑾峰彇褰撳墠妯″瀷闇€瑕佺殑杈撳叆闀垮害 (渚嬪浠?512 闄嶅埌浜?400)
target_len = input_details[0]['shape'][1]
print(f"TFLite expected input shape: {input_details[0]['shape']}")

# ====================== 3. 鍔犺浇骞跺姩鎬侀檷閲囨牱鏁版嵁 ======================
noisy = np.load(noisy_path)

if noisy.ndim == 1:
    noisy = noisy.reshape(1, -1)
elif noisy.ndim == 3 and noisy.shape[-1] == 1:
    noisy = noisy[..., 0] 

orig_len = noisy.shape[1]
N = noisy.shape[0]

# ---> 鏍稿績锛氬鏋滀綘璇诲彇鐨勬暟鎹暱搴﹀拰妯″瀷涓嶅尮閰嶏紝鑷姩璋冪敤 scipy 杩涜闄嶉噰鏍?<---
if orig_len != target_len:
    print(f"Data length mismatch! Resampling from {orig_len} to {target_len} points...")
    noisy = resample_1d_batch(noisy, target_len)
else:
    print("Data length matches model. No resampling needed.")

print(f"Ready for inference. Using N = {N}, segment_len = {target_len}")

# ====================== 4. 寮€濮嬪崟鏍锋湰鎺ㄧ悊娴嬭瘯 ======================
total_inference_time = 0.0
single_segment_inference_times = []
results = []

print("\n==== Start inference only ====")
start_all = time.time()

for i in range(N):
    segment_1d = noisy[i].astype(np.float32)
    
    # 閫傞厤 fcNN 鐨勮緭鍏ュ舰鐘? (1, target_len)
    segment_reshaped = segment_1d.reshape(1, target_len)
    interpreter.set_tensor(input_details[0]['index'], segment_reshaped)

    # ---------- 绾帹鐞嗚鏃跺紑濮?----------
    t0 = time.time()
    interpreter.invoke()
    t1 = time.time()
    # ---------- 绾帹鐞嗚鏃剁粨鏉?----------

    dt = t1 - t0
    total_inference_time += dt
    single_segment_inference_times.append(dt)

    out = interpreter.get_tensor(output_details[0]['index'])
    results.append(np.squeeze(out))

end_all = time.time()
wall_clock_time = end_all - start_all

# ====================== 5. 鎵撳嵃鏋侀檺娴嬭瘯鎸囨爣 ======================
single_segment_inference_times = np.array(single_segment_inference_times, dtype=np.float64)

avg_single = single_segment_inference_times.mean()
min_single = single_segment_inference_times.min()
max_single = single_segment_inference_times.max()

print("\n==== Inference time (invoke only) ====")
print(f"Total invoke time: {total_inference_time:.6f} s")
print(f"Average per segment: {avg_single:.6f} s ({avg_single * 1000.0:.2f} ms)")
print(f"Min / Max per segment: {min_single:.6f} s / {max_single:.6f} s")
print(f"First segment: {single_segment_inference_times[0]:.6f} s ({single_segment_inference_times[0] * 1000.0:.2f} ms)")
print(f"\nWall-clock time (including loop overhead): {wall_clock_time:.6f} s")
print("Inference only finished.")
