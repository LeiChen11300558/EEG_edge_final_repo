# -*- coding: utf-8 -*-
import numpy as np
import os
import tflite_runtime.interpreter as tflite

NUM_THREADS = 4

# ====================== 1. Math & Metrics ======================
def rms_value(arr):
    arr = np.asarray(arr, dtype=np.float32)
    return float(np.sqrt(np.mean(arr ** 2)))

def RRMSE(true, pred):
    num = rms_value(true - pred)
    den = rms_value(true)
    if den == 0: return 0.0
    return float(num / den)

def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()
    x_m, y_m = np.mean(x), np.mean(y)
    num = np.sum((x - x_m) * (y - y_m))
    den = np.sqrt(np.sum((x - x_m)**2) * np.sum((y - y_m)**2))
    if den == 0: return 0.0
    return float(num / den)

def resample_np(data_2d, target_len):
    orig_len = data_2d.shape[1]
    if orig_len == target_len: return data_2d
    N = data_2d.shape[0]
    x_orig = np.linspace(0, 1, orig_len)
    x_target = np.linspace(0, 1, target_len)
    new_data = np.zeros((N, target_len), dtype=np.float32)
    for i in range(N):
        new_data[i] = np.interp(x_target, x_orig, data_2d[i])
    return new_data

# ====================== 2. Paths & Config ======================
current_path = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "fu200hz"   
data_dir = os.path.join(current_path, "..", "data")

model_path = os.path.join(current_path, MODEL_NAME)
noisy_path = os.path.join(data_dir, "noiseinput_test.npy")
clean_path = os.path.join(data_dir, "EEG_test.npy")

# ====================== 3. Load & Process Data ======================
print(f"Creating TFLite interpreter with num_threads = {NUM_THREADS}")
interpreter = tflite.Interpreter(model_path=model_path, num_threads=NUM_THREADS)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
target_len = input_details[0]['shape'][1]

print("Loading data...")
noisy = np.load(noisy_path)
clean = np.load(clean_path)

if noisy.ndim == 3: noisy = noisy[..., 0]
if clean.ndim == 3: clean = clean[..., 0]

print(f"Resampling data to {target_len} points using pure Numpy...")
noisy_200hz = resample_np(noisy, target_len)
clean_200hz = resample_np(clean, target_len)
N = noisy_200hz.shape[0]

# ====================== 4. Run Inference ======================
print("Running inference...")
denoised_results = []
for i in range(N):
    sample = noisy_200hz[i].astype(np.float32).reshape(1, target_len)
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    denoised_results.append(np.squeeze(out))

denoised_200hz = np.array(denoised_results, dtype=np.float32)

# ====================== 5. Evaluate & Print ======================
noisy_rrmse, noisy_cc = [], []
deno_rrmse, deno_cc = [], []

for i in range(N):
    c = clean_200hz[i]
    n = noisy_200hz[i]
    d = denoised_200hz[i]
    
    noisy_rrmse.append(RRMSE(c, n))
    noisy_cc.append(pearson_corr(c, n))
    
    deno_rrmse.append(RRMSE(c, d))
    deno_cc.append(pearson_corr(c, d))

print("\n" + "="*45)
print(" Denoising Performance Report (Time-Domain)")
print("="*45)
print("[Before Denoising (Noisy vs. Clean)]:")
print(f"  RRMSE: {np.mean(noisy_rrmse):.4f}  (Lower is better)")
print(f"  CC   : {np.mean(noisy_cc):.4f}  (Closer to 1.0 is better)")
print("-" * 45)
print("[After Denoising (Denoised vs. Clean)]:")
print(f"  RRMSE: {np.mean(deno_rrmse):.4f}  (Lower is better)")
print(f"  CC   : {np.mean(deno_cc):.4f}  (Closer to 1.0 is better)")
print("="*45)
print("Evaluation finished.")
