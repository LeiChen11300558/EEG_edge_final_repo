# -*- coding: utf-8 -*-
import math
import os
import time

import numpy as np
import tflite_runtime.interpreter as tflite


NUM_THREADS = 4
MODEL_NAME = "fu8_large_200hz"
NOISY_NAME = "noiseinput_test.npy"
CLEAN_NAME = "EEG_test.npy"


def resolve_model_path(path_without_ext):
    if os.path.exists(path_without_ext):
        return path_without_ext
    if os.path.exists(path_without_ext + ".tflite"):
        return path_without_ext + ".tflite"
    raise FileNotFoundError(f"Model not found: {path_without_ext} or {path_without_ext}.tflite")


def to_3d(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1, 1)
    if arr.ndim == 2:
        return arr[..., np.newaxis]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unsupported data shape: {arr.shape}")


def resample_1d(signal_1d, target_len):
    signal_1d = np.asarray(signal_1d, dtype=np.float32).reshape(-1)
    if signal_1d.size == target_len:
        return signal_1d
    old_x = np.linspace(0.0, 1.0, signal_1d.size, dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.interp(new_x, old_x, signal_1d).astype(np.float32)


def resample_sample(sample, target_len):
    sample = np.asarray(sample, dtype=np.float32)
    if sample.ndim == 1:
        return resample_1d(sample, target_len)
    channels = [resample_1d(sample[:, ch], target_len) for ch in range(sample.shape[1])]
    return np.stack(channels, axis=-1).astype(np.float32)


def rms_value(arr):
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    return math.sqrt(float(np.mean(arr ** 2)))


def rrmse(true, pred):
    den = rms_value(true)
    if den == 0.0:
        return 0.0
    return rms_value(np.asarray(true) - np.asarray(pred)) / den


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size == 0:
        return 0.0
    x = x - np.mean(x)
    y = y - np.mean(y)
    den = math.sqrt(float(np.sum(x ** 2) * np.sum(y ** 2)))
    if den == 0.0:
        return 0.0
    return float(np.sum(x * y)) / den


def calc_metrics_nd(clean, noisy, denoised):
    clean = np.asarray(clean, dtype=np.float32) - np.mean(clean)
    noisy = np.asarray(noisy, dtype=np.float32) - np.mean(noisy)
    denoised = np.asarray(denoised, dtype=np.float32) - np.mean(denoised)

    if clean.ndim == 1:
        clean_ch = clean.reshape(-1, 1)
        noisy_ch = noisy.reshape(-1, 1)
        deno_ch = denoised.reshape(-1, 1)
    else:
        clean_ch = clean
        noisy_ch = noisy
        deno_ch = denoised

    channel_cc = [pearson_corr(clean_ch[:, ch], deno_ch[:, ch]) for ch in range(clean_ch.shape[1])]
    baseline_channel_cc = [pearson_corr(clean_ch[:, ch], noisy_ch[:, ch]) for ch in range(clean_ch.shape[1])]

    return {
        "rrmse": rrmse(clean, denoised),
        "baseline_rrmse": rrmse(clean, noisy),
        "pearson_cc_global": pearson_corr(clean, denoised),
        "baseline_pearson_cc_global": pearson_corr(clean, noisy),
        "pearson_cc_channel_mean": float(np.mean(channel_cc)),
        "pearson_cc_channel_min": float(np.min(channel_cc)),
        "pearson_cc_channel_max": float(np.max(channel_cc)),
        "baseline_pearson_cc_channel_mean": float(np.mean(baseline_channel_cc)),
        "baseline_pearson_cc_channel_min": float(np.min(baseline_channel_cc)),
        "baseline_pearson_cc_channel_max": float(np.max(baseline_channel_cc)),
    }


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_path, "..", "data")
    model_path = resolve_model_path(os.path.join(current_path, MODEL_NAME))
    noisy_path = os.path.join(data_dir, NOISY_NAME)
    clean_path = os.path.join(data_dir, CLEAN_NAME)

    print("Model path:", model_path)
    print("Noisy data path:", noisy_path)
    print("Clean data path:", clean_path)
    print(f"Creating TFLite interpreter with num_threads = {NUM_THREADS}")

    interpreter = tflite.Interpreter(model_path=model_path, num_threads=NUM_THREADS)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    target_len = int(input_shape[1])
    target_ch = int(input_shape[2])
    print("TFLite expected input shape:", input_shape)

    noisy = to_3d(np.load(noisy_path))
    clean = to_3d(np.load(clean_path))
    N = min(noisy.shape[0], clean.shape[0])
    noisy = noisy[:N]
    clean = clean[:N]

    if noisy.shape[1] != target_len:
        print(f"Data length mismatch! Resampling from {noisy.shape[1]} to {target_len} points...")
        noisy = np.stack([resample_sample(x, target_len) for x in noisy], axis=0)
        clean = np.stack([resample_sample(x, target_len) for x in clean], axis=0)
    else:
        print("Data length matches model. No resampling needed.")

    if noisy.shape[2] != target_ch:
        raise ValueError(f"Channel mismatch: data has {noisy.shape[2]} channels, model expects {target_ch}")

    print(f"Ready for inference. N = {N}, segment_len = {target_len}, channels = {target_ch}")

    total_inference_time = 0.0
    single_times = []
    decoded = []

    print("\n==== Start inference only ====")
    for i in range(N):
        sample = noisy[i].astype(np.float32).reshape(1, target_len, target_ch)
        interpreter.set_tensor(input_details[0]["index"], sample)
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()
        dt = t1 - t0
        total_inference_time += dt
        single_times.append(dt)
        out = interpreter.get_tensor(output_details[0]["index"])
        decoded.append(np.squeeze(out))

    decoded = np.asarray(decoded, dtype=np.float32)
    single_times = np.asarray(single_times, dtype=np.float64)

    rrmse_vals = []
    base_rrmse_vals = []
    cc_global_vals = []
    base_cc_global_vals = []
    cc_ch_mean_vals = []
    cc_ch_min_vals = []
    cc_ch_max_vals = []
    base_cc_ch_mean_vals = []
    base_cc_ch_min_vals = []
    base_cc_ch_max_vals = []

    for i in range(N):
        m = calc_metrics_nd(clean[i], noisy[i], decoded[i])
        rrmse_vals.append(m["rrmse"])
        base_rrmse_vals.append(m["baseline_rrmse"])
        cc_global_vals.append(m["pearson_cc_global"])
        base_cc_global_vals.append(m["baseline_pearson_cc_global"])
        cc_ch_mean_vals.append(m["pearson_cc_channel_mean"])
        cc_ch_min_vals.append(m["pearson_cc_channel_min"])
        cc_ch_max_vals.append(m["pearson_cc_channel_max"])
        base_cc_ch_mean_vals.append(m["baseline_pearson_cc_channel_mean"])
        base_cc_ch_min_vals.append(m["baseline_pearson_cc_channel_min"])
        base_cc_ch_max_vals.append(m["baseline_pearson_cc_channel_max"])

    print("\n==== Inference time (invoke only) ====")
    print(f"Total invoke time: {total_inference_time:.6f} s")
    print(f"Average per segment: {single_times.mean():.6f} s ({single_times.mean() * 1000.0:.2f} ms)")
    print(f"Min / Max per segment: {single_times.min():.6f} s / {single_times.max():.6f} s")

    print("\n==== Denoising Performance Report (8-channel) ====")
    print("[Before Denoising (Noisy vs. Clean)]:")
    print(f"  RRMSE mean                 : {np.mean(base_rrmse_vals):.4f}")
    print(f"  Pearson CC global mean     : {np.mean(base_cc_global_vals):.4f}")
    print(f"  Pearson CC channel mean    : {np.mean(base_cc_ch_mean_vals):.4f}")
    print(f"  Pearson CC channel min mean: {np.mean(base_cc_ch_min_vals):.4f}")
    print(f"  Pearson CC channel max mean: {np.mean(base_cc_ch_max_vals):.4f}")
    print("-" * 48)
    print("[After Denoising (Denoised vs. Clean)]:")
    print(f"  RRMSE mean                 : {np.mean(rrmse_vals):.4f}")
    print(f"  Pearson CC global mean     : {np.mean(cc_global_vals):.4f}")
    print(f"  Pearson CC channel mean    : {np.mean(cc_ch_mean_vals):.4f}")
    print(f"  Pearson CC channel min mean: {np.mean(cc_ch_min_vals):.4f}")
    print(f"  Pearson CC channel max mean: {np.mean(cc_ch_max_vals):.4f}")
    print("=" * 48)
    print("Evaluation finished.")
