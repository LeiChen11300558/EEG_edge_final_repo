# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import math
import tflite_runtime.interpreter as tflite

NUM_THREADS = 4

# ====================== 1. Paths ======================
current_path = os.path.dirname(os.path.abspath(__file__))

# TFLite model file name
MODEL_NAME = "fu"   # 如果是 fu.tflite，就写 "fu.tflite"

# data directory (../data)
data_dir = os.path.join(current_path, "..", "data")

NOISY_NAME = "noiseinput_test.npy"
CLEAN_NAME = "EEG_test.npy"

model_path = os.path.join(current_path, MODEL_NAME)
noisy_path = os.path.join(data_dir, NOISY_NAME)
clean_path = os.path.join(data_dir, CLEAN_NAME)

print("Model path:", model_path)
print("Noisy  data path:", noisy_path)
print("Clean  data path:", clean_path)

# ====================== 2. Load data ======================
noisy = np.load(noisy_path)
clean = np.load(clean_path)

print("Noisy  shape:", noisy.shape, ", dtype:", noisy.dtype)
print("Clean  shape:", clean.shape, ", dtype:", clean.dtype)

# Make sure same number of samples
N = min(noisy.shape[0], clean.shape[0])
noisy = noisy[:N]
clean = clean[:N]

def to_2d(arr):
    """Convert (N,L,1) or (N,L) or (L,) to (N,L) float32"""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    return arr

noisy_2d = to_2d(noisy)
clean_2d = to_2d(clean)

N, segment_len = noisy_2d.shape
print("Using N =", N, ", segment_len =", segment_len)

# ====================== 3. Load TFLite model ======================
print(f"Creating TFLite interpreter with num_threads = {NUM_THREADS}")
interpreter = tflite.Interpreter(model_path=model_path, num_threads=NUM_THREADS)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite input_details:", input_details)
print("TFLite output_details:", output_details)

input_shape = input_details[0]['shape']  # e.g. [1, 512, 1]

# ====================== 4. Inference + timing ======================
total_inference_time = 0.0
single_segment_inference_times = []
results = []

start_all = time.time()

for segment in noisy_2d:
    # segment: (L,) -> (1, L, 1) according to input_shape
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

    out = interpreter.get_tensor(output_details[0]['index'])  # (1, L) or (1,L,1)
    out = np.squeeze(out)                                     # (L,)
    results.append(out)

end_all = time.time()
wall_clock_time = end_all - start_all

results = np.array(results, dtype=np.float32)   # (N, L)
decoded_2d = results.reshape(N, segment_len)

print("decoded_layer shape:", decoded_2d.shape)

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

# ====================== 5. Zero-centering ======================
# For fair comparison we always zero-center per segment
z_clean   = clean_2d  - np.mean(clean_2d,  axis=1, keepdims=True)
z_noisy   = noisy_2d  - np.mean(noisy_2d,  axis=1, keepdims=True)
z_decoded = decoded_2d - np.mean(decoded_2d, axis=1, keepdims=True)

# ====================== 6. Metric functions ======================
def rms_value(arr):
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    square = float(np.sum(arr ** 2))
    mean = square / float(arr.size)
    return math.sqrt(mean)

def RRMSE(true, pred):
    num = rms_value(true - pred)
    den = rms_value(true)
    if den == 0.0:
        return 0.0
    return num / den

def RMSE(true, pred):
    return rms_value(true - pred)

def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size == 0:
        return 0.0
    xm = x - x.mean()
    ym = y - y.mean()
    num = float(np.sum(xm * ym))
    den = math.sqrt(float(np.sum(xm ** 2) * np.sum(ym ** 2)))
    if den == 0.0:
        return 0.0
    return num / den

def welch_psd(signal_1d, fs=200.0, nperseg=200, nfft=512):
    """
    Simple Welch PSD estimate using only numpy.
    """
    x = np.asarray(signal_1d, dtype=np.float64).ravel()
    if x.size == 0:
        freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
        return freqs, np.zeros_like(freqs, dtype=np.float64)

    if nperseg > x.size:
        nperseg = x.size
    step = max(1, nperseg // 2)  # 50% overlap

    window = np.hanning(nperseg).astype(np.float64)
    scale = fs * float(np.sum(window ** 2))

    seg_psds = []
    for start in range(0, x.size - nperseg + 1, step):
        seg = x[start:start + nperseg] * window
        spec = np.fft.rfft(seg, n=nfft)
        psd = (np.abs(spec) ** 2) / scale
        seg_psds.append(psd)

    if not seg_psds:
        seg = x[:nperseg] * window
        spec = np.fft.rfft(seg, n=nfft)
        psd = (np.abs(spec) ** 2) / scale
        seg_psds = [psd]

    seg_psds = np.stack(seg_psds, axis=0)
    mean_psd = np.mean(seg_psds, axis=0)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return freqs, mean_psd

# ====================== 7. Time-domain metrics ======================
# decoded vs clean (真正的去噪效果)
time_rrmse_denoised = []
time_rmse_denoised  = []
cc_denoised         = []

# noisy vs clean (baseline)
time_rrmse_noisy = []
time_rmse_noisy  = []
cc_noisy         = []

for i in range(N):
    # denoised vs clean
    time_rrmse_denoised.append(RRMSE(z_clean[i], z_decoded[i]))
    time_rmse_denoised.append(RMSE(z_clean[i], z_decoded[i]))
    cc_denoised.append(pearson_corr(z_clean[i], z_decoded[i]))

    # noisy vs clean
    time_rrmse_noisy.append(RRMSE(z_clean[i], z_noisy[i]))
    time_rmse_noisy.append(RMSE(z_clean[i], z_noisy[i]))
    cc_noisy.append(pearson_corr(z_clean[i], z_noisy[i]))

time_rrmse_denoised = np.array(time_rrmse_denoised, dtype=np.float64)
time_rmse_denoised  = np.array(time_rmse_denoised,  dtype=np.float64)
cc_denoised         = np.array(cc_denoised,         dtype=np.float64)

time_rrmse_noisy = np.array(time_rrmse_noisy, dtype=np.float64)
time_rmse_noisy  = np.array(time_rmse_noisy,  dtype=np.float64)
cc_noisy         = np.array(cc_noisy,         dtype=np.float64)

# ====================== 8. Frequency-domain metrics ======================
nfft = segment_len
PSD_len = nfft // 2 + 1

clean_psd   = np.zeros((N, PSD_len), dtype=np.float64)
decoded_psd = np.zeros((N, PSD_len), dtype=np.float64)
noisy_psd   = np.zeros((N, PSD_len), dtype=np.float64)

for i in range(N):
    _, pxx_clean   = welch_psd(z_clean[i],   fs=200.0, nperseg=min(200, segment_len), nfft=nfft)
    _, pxx_decoded = welch_psd(z_decoded[i], fs=200.0, nperseg=min(200, segment_len), nfft=nfft)
    _, pxx_noisy   = welch_psd(z_noisy[i],   fs=200.0, nperseg=min(200, segment_len), nfft=nfft)

    clean_psd[i]   = pxx_clean
    decoded_psd[i] = pxx_decoded
    noisy_psd[i]   = pxx_noisy

freq_rrmse_denoised = []
freq_rmse_denoised  = []
freq_rrmse_noisy    = []
freq_rmse_noisy     = []

for i in range(N):
    # denoised vs clean
    freq_rrmse_denoised.append(RRMSE(clean_psd[i], decoded_psd[i]))
    freq_rmse_denoised.append(RMSE(clean_psd[i], decoded_psd[i]))
    # noisy vs clean (baseline)
    freq_rrmse_noisy.append(RRMSE(clean_psd[i], noisy_psd[i]))
    freq_rmse_noisy.append(RMSE(clean_psd[i], noisy_psd[i]))

freq_rrmse_denoised = np.array(freq_rrmse_denoised, dtype=np.float64)
freq_rmse_denoised  = np.array(freq_rmse_denoised,  dtype=np.float64)
freq_rrmse_noisy    = np.array(freq_rrmse_noisy,    dtype=np.float64)
freq_rmse_noisy     = np.array(freq_rmse_noisy,     dtype=np.float64)

# ====================== 9. Print metrics ======================
print("\n==== Denoising metrics (clean vs. decoded) ====")
print("Time-domain RRMSE:  mean = {:.4f}, std = {:.4f}".format(
    time_rrmse_denoised.mean(), time_rrmse_denoised.std()))
print("Time-domain RMSE:   mean = {:.4f}, std = {:.4f}".format(
    time_rmse_denoised.mean(), time_rmse_denoised.std()))
print("Freq-domain RRMSE:  mean = {:.4f}, std = {:.4f}".format(
    freq_rrmse_denoised.mean(), freq_rrmse_denoised.std()))
print("Freq-domain RMSE:   mean = {:.4f}, std = {:.4f}".format(
    freq_rmse_denoised.mean(), freq_rmse_denoised.std()))
print("Pearson CC:         mean = {:.4f}, std = {:.4f}".format(
    cc_denoised.mean(), cc_denoised.std()))

print("\n==== Baseline metrics (clean vs. noisy input) ====")
print("Time-domain RRMSE:  mean = {:.4f}, std = {:.4f}".format(
    time_rrmse_noisy.mean(), time_rrmse_noisy.std()))
print("Time-domain RMSE:   mean = {:.4f}, std = {:.4f}".format(
    time_rmse_noisy.mean(), time_rmse_noisy.std()))
print("Freq-domain RRMSE:  mean = {:.4f}, std = {:.4f}".format(
    freq_rrmse_noisy.mean(), freq_rrmse_noisy.std()))
print("Freq-domain RMSE:   mean = {:.4f}, std = {:.4f}".format(
    freq_rmse_noisy.mean(), freq_rmse_noisy.std()))
print("Pearson CC:         mean = {:.4f}, std = {:.4f}".format(
    cc_noisy.mean(), cc_noisy.std()))
