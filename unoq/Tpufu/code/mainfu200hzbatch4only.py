# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import tflite_runtime.interpreter as tflite

NUM_THREADS = 4
BATCH_SIZE = 4
MODEL_NAME = "fu200hzbatch4"
NOISY_NAME = "noiseinput_test.npy"

current_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_path, "..", "data")
model_path = os.path.join(current_path, MODEL_NAME)
noisy_path = os.path.join(data_dir, NOISY_NAME)


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


if __name__ == "__main__":
    resolved_model_path = resolve_model_path(model_path)
    print("Model path:", resolved_model_path)
    print("Noisy data path:", noisy_path)
    print(f"Creating TFLite interpreter with num_threads = {NUM_THREADS}")

    interpreter = tflite.Interpreter(model_path=resolved_model_path, num_threads=NUM_THREADS)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    target_len = int(input_details[0]['shape'][1])
    print("TFLite expected input shape:", input_details[0]['shape'])

    noisy = to_2d(np.load(noisy_path))
    if noisy.shape[1] != target_len:
        print(f"Data length mismatch! Resampling from {noisy.shape[1]} to {target_len} points...")
        noisy = resample_1d_batch(noisy, target_len)
    else:
        print("Data length matches model. No resampling needed.")

    noisy_padded, pad_count = pad_to_batch(noisy, BATCH_SIZE)
    num_batches = len(noisy_padded) // BATCH_SIZE
    print(f"Ready for batch inference. N = {len(noisy)}, batch_size = {BATCH_SIZE}, num_batches = {num_batches}, segment_len = {target_len}")

    batch_times = []
    results = []
    print("\n==== Start batch-4 inference only ====")
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

    batch_times = np.asarray(batch_times, dtype=np.float64)
    per_segment_times = batch_times / float(BATCH_SIZE)

    print("\n==== Inference time (invoke only) ====")
    print(f"Total invoke time: {batch_times.sum():.6f} s")
    print(f"Average per batch: {batch_times.mean():.6f} s ({batch_times.mean() * 1000.0:.2f} ms)")
    print(f"Average per segment: {per_segment_times.mean():.6f} s ({per_segment_times.mean() * 1000.0:.2f} ms)")
    print(f"Min / Max per batch: {batch_times.min():.6f} s / {batch_times.max():.6f} s")
    print(f"Wall-clock time (including loop overhead): {wall_clock_time:.6f} s")
    print(f"Decoded output shape: {decoded.shape}")
    print("Inference only finished.")
