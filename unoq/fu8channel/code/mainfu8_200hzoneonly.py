# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import tflite_runtime.interpreter as tflite


NUM_THREADS = 4
MODEL_NAME = "fu8_large_200hz"
NOISY_NAME = "noiseinput_test.npy"
SAMPLE_INDEX = 0
WARMUP_RUNS = 1
MEASURE_RUNS = 5


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


if __name__ == "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_path, "..", "data")

    model_path = resolve_model_path(os.path.join(current_path, MODEL_NAME))
    noisy_path = os.path.join(data_dir, NOISY_NAME)

    print("Model path:", model_path)
    print("Noisy data path:", noisy_path)
    print(f"Creating TFLite interpreter with num_threads = {NUM_THREADS}")

    interpreter = tflite.Interpreter(model_path=model_path, num_threads=NUM_THREADS)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    target_len = int(input_details[0]["shape"][1])
    target_ch = int(input_details[0]["shape"][2])
    print("TFLite expected input shape:", input_details[0]["shape"])

    noisy = to_3d(np.load(noisy_path))
    if noisy.shape[1] != target_len:
        print(f"Data length mismatch! Resampling from {noisy.shape[1]} to {target_len} points...")
        noisy = np.stack([resample_sample(x, target_len) for x in noisy], axis=0)
    else:
        print("Data length matches model. No resampling needed.")

    if noisy.shape[2] != target_ch:
        raise ValueError(f"Channel mismatch: data has {noisy.shape[2]} channels, model expects {target_ch}")
    if SAMPLE_INDEX < 0 or SAMPLE_INDEX >= len(noisy):
        raise IndexError(f"SAMPLE_INDEX={SAMPLE_INDEX} out of range for N={len(noisy)}")

    sample = noisy[SAMPLE_INDEX].astype(np.float32).reshape(1, target_len, target_ch)
    print(f"Using sample index = {SAMPLE_INDEX}")
    print(f"Sample shape = {sample.shape}")

    print("\n==== Warmup ====")
    for i in range(WARMUP_RUNS):
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        print(f"Warmup {i + 1}/{WARMUP_RUNS} done")

    print("\n==== Timed runs (inference only) ====")
    times = []
    for i in range(MEASURE_RUNS):
        interpreter.set_tensor(input_details[0]["index"], sample)
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()
        dt = t1 - t0
        times.append(dt)
        print(f"Run {i + 1}/{MEASURE_RUNS}: {dt:.6f} s ({dt * 1000.0:.2f} ms)")

    times = np.asarray(times, dtype=np.float64)
    print("\n==== Summary ====")
    print(f"Average: {times.mean():.6f} s ({times.mean() * 1000.0:.2f} ms)")
    print(f"Min:     {times.min():.6f} s ({times.min() * 1000.0:.2f} ms)")
    print(f"Max:     {times.max():.6f} s ({times.max() * 1000.0:.2f} ms)")
    print("Inference-only test finished.")
