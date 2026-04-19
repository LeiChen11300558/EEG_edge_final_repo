# -*- coding: utf-8 -*-
import math
import os
import time
import numpy as np
import tflite_runtime.interpreter as tflite

NUM_THREADS = 4
BATCH_SIZE = 4
MODEL_NAME = "fu200hzbatch4"
NOISY_NAME = "noiseinput_test.npy"
CLEAN_NAME = "EEG_test.npy"
FS = 200.0

current_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_path, "..", "data")
model_path = os.path.join(current_path, MODEL_NAME)
noisy_path = os.path.join(data_dir, NOISY_NAME)
clean_path = os.path.join(data_dir, CLEAN_NAME)


def resolve_model_path(path_without_ext):
    if os.path.exists(path_without_ext):
        return path_without_ext
    if os.path.exists(path_without_ext + ".tflite"):
        return path_without_ext + ".tflite"
    raise FileNotFoundError(f"Model not found: {path_without_ext} or {path_without_ext}.tflite")


def to_2d(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


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


def pad_to_batch(arr, batch_size):
    remainder = len(arr) % batch_size
    if remainder == 0:
        return arr, 0
    pad = batch_size - remainder
    padded = np.concatenate([arr, np.repeat(arr[-1:, :], pad, axis=0)], axis=0)
    return padded, pad


def rms_value(arr):
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    return math.sqrt(float(np.sum(arr ** 2)) / float(arr.size))


def rrmse(true, pred):
    den = rms_value(true)
    if den == 0.0:
        return 0.0
    return rms_value(true - pred) / den


def rmse(true, pred):
    return rms_value(true - pred)


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size == 0:
        return 0.0
    xm = x - x.mean()
    ym = y - y.mean()
    den = math.sqrt(float(np.sum(xm ** 2) * np.sum(ym ** 2)))
    if den == 0.0:
        return 0.0
    return float(np.sum(xm * ym)) / den


def welch_psd(signal_1d, fs=200.0, nperseg=200, nfft=512):
    x = np.asarray(signal_1d, dtype=np.float64).ravel()
    if x.size == 0:
        freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
        return freqs, np.zeros_like(freqs, dtype=np.float64)
    if nperseg > x.size:
        nperseg = x.size
    step = max(1, nperseg // 2)
    window = np.hanning(nperseg).astype(np.float64)
    scale = fs * float(np.sum(window ** 2))
    seg_psds = []
    for start in range(0, x.size - nperseg + 1, step):
        seg = x[start:start + nperseg] * window
        spec = np.fft.rfft(seg, n=nfft)
        seg_psds.append((np.abs(spec) ** 2) / scale)
    if not seg_psds:
        seg = x[:nperseg] * window
        spec = np.fft.rfft(seg, n=nfft)
        seg_psds = [(np.abs(spec) ** 2) / scale]
    mean_psd = np.mean(np.stack(seg_psds, axis=0), axis=0)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return freqs, mean_psd


if __name__ == "__main__":
    resolved_model_path = resolve_model_path(model_path)
    print("Model path:", resolved_model_path)
    print("Noisy data path:", noisy_path)
    print("Clean data path:", clean_path)
    print(f"Creating TFLite interpreter with num_threads = {NUM_THREADS}")

    interpreter = tflite.Interpreter(model_path=resolved_model_path, num_threads=NUM_THREADS)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    target_len = int(input_details[0]['shape'][1])
    print("TFLite expected input shape:", input_details[0]['shape'])

    noisy = to_2d(np.load(noisy_path))
    clean = to_2d(np.load(clean_path))
    N = min(noisy.shape[0], clean.shape[0])
    noisy = noisy[:N]
    clean = clean[:N]

    if noisy.shape[1] != target_len:
        print(f"Data length mismatch! Resampling from {noisy.shape[1]} to {target_len} points...")
        noisy = resample_1d_batch(noisy, target_len)
        clean = resample_1d_batch(clean, target_len)
    else:
        print("Data length matches model. No resampling needed.")

    noisy_padded, pad_count = pad_to_batch(noisy, BATCH_SIZE)
    clean_padded, _ = pad_to_batch(clean, BATCH_SIZE)
    num_batches = len(noisy_padded) // BATCH_SIZE
    print(f"Ready for batch inference. N = {len(noisy)}, batch_size = {BATCH_SIZE}, num_batches = {num_batches}, segment_len = {target_len}")

    batch_times = []
    results = []
    print("\n==== Start batch-4 inference ====")
    start_all = time.time()

    for b in range(num_batches):
        batch = noisy_padded[b * BATCH_SIZE:(b + 1) * BATCH_SIZE].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], batch)
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()
        batch_times.append(t1 - t0)
        out = interpreter.get_tensor(output_details[0]['index'])
        results.append(np.asarray(out, dtype=np.float32))

    end_all = time.time()
    wall_clock_time = end_all - start_all

    decoded = np.concatenate(results, axis=0)
    if pad_count > 0:
        decoded = decoded[:-pad_count]
    clean = clean_padded[:len(decoded)]
    noisy = noisy_padded[:len(decoded)]

    batch_times = np.asarray(batch_times, dtype=np.float64)
    per_segment_times = batch_times / float(BATCH_SIZE)

    print("\n==== Inference time (invoke only) ====")
    print(f"Total invoke time: {batch_times.sum():.6f} s")
    print(f"Average per batch: {batch_times.mean():.6f} s ({batch_times.mean() * 1000.0:.2f} ms)")
    print(f"Average per segment: {per_segment_times.mean():.6f} s ({per_segment_times.mean() * 1000.0:.2f} ms)")
    print(f"Min / Max per batch: {batch_times.min():.6f} s / {batch_times.max():.6f} s")
    print(f"Wall-clock time (including loop overhead): {wall_clock_time:.6f} s")

    z_clean = clean - np.mean(clean, axis=1, keepdims=True)
    z_noisy = noisy - np.mean(noisy, axis=1, keepdims=True)
    z_decoded = decoded - np.mean(decoded, axis=1, keepdims=True)

    time_rrmse_denoised = np.asarray([rrmse(z_clean[i], z_decoded[i]) for i in range(len(decoded))], dtype=np.float64)
    time_rmse_denoised = np.asarray([rmse(z_clean[i], z_decoded[i]) for i in range(len(decoded))], dtype=np.float64)
    cc_denoised = np.asarray([pearson_corr(z_clean[i], z_decoded[i]) for i in range(len(decoded))], dtype=np.float64)
    time_rrmse_noisy = np.asarray([rrmse(z_clean[i], z_noisy[i]) for i in range(len(decoded))], dtype=np.float64)
    time_rmse_noisy = np.asarray([rmse(z_clean[i], z_noisy[i]) for i in range(len(decoded))], dtype=np.float64)
    cc_noisy = np.asarray([pearson_corr(z_clean[i], z_noisy[i]) for i in range(len(decoded))], dtype=np.float64)

    nfft = target_len
    psd_len = nfft // 2 + 1
    clean_psd = np.zeros((len(decoded), psd_len), dtype=np.float64)
    decoded_psd = np.zeros((len(decoded), psd_len), dtype=np.float64)
    noisy_psd = np.zeros((len(decoded), psd_len), dtype=np.float64)
    for i in range(len(decoded)):
        _, clean_psd[i] = welch_psd(z_clean[i], fs=FS, nperseg=min(200, target_len), nfft=nfft)
        _, decoded_psd[i] = welch_psd(z_decoded[i], fs=FS, nperseg=min(200, target_len), nfft=nfft)
        _, noisy_psd[i] = welch_psd(z_noisy[i], fs=FS, nperseg=min(200, target_len), nfft=nfft)

    freq_rrmse_denoised = np.asarray([rrmse(clean_psd[i], decoded_psd[i]) for i in range(len(decoded))], dtype=np.float64)
    freq_rmse_denoised = np.asarray([rmse(clean_psd[i], decoded_psd[i]) for i in range(len(decoded))], dtype=np.float64)
    freq_rrmse_noisy = np.asarray([rrmse(clean_psd[i], noisy_psd[i]) for i in range(len(decoded))], dtype=np.float64)
    freq_rmse_noisy = np.asarray([rmse(clean_psd[i], noisy_psd[i]) for i in range(len(decoded))], dtype=np.float64)

    print("\n==== Denoising metrics (clean vs. decoded) ====")
    print("Time-domain RRMSE:  mean = {:.4f}, std = {:.4f}".format(time_rrmse_denoised.mean(), time_rrmse_denoised.std()))
    print("Time-domain RMSE:   mean = {:.4f}, std = {:.4f}".format(time_rmse_denoised.mean(), time_rmse_denoised.std()))
    print("Freq-domain RRMSE:  mean = {:.4f}, std = {:.4f}".format(freq_rrmse_denoised.mean(), freq_rrmse_denoised.std()))
    print("Freq-domain RMSE:   mean = {:.4f}, std = {:.4f}".format(freq_rmse_denoised.mean(), freq_rmse_denoised.std()))
    print("Pearson CC:         mean = {:.4f}, std = {:.4f}".format(cc_denoised.mean(), cc_denoised.std()))

    print("\n==== Baseline metrics (clean vs. noisy input) ====")
    print("Time-domain RRMSE:  mean = {:.4f}, std = {:.4f}".format(time_rrmse_noisy.mean(), time_rrmse_noisy.std()))
    print("Time-domain RMSE:   mean = {:.4f}, std = {:.4f}".format(time_rmse_noisy.mean(), time_rmse_noisy.std()))
    print("Freq-domain RRMSE:  mean = {:.4f}, std = {:.4f}".format(freq_rrmse_noisy.mean(), freq_rrmse_noisy.std()))
    print("Freq-domain RMSE:   mean = {:.4f}, std = {:.4f}".format(freq_rmse_noisy.mean(), freq_rmse_noisy.std()))
    print("Pearson CC:         mean = {:.4f}, std = {:.4f}".format(cc_noisy.mean(), cc_noisy.std()))
