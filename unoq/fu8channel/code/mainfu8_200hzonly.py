# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import tflite_runtime.interpreter as tflite


SAVE_OUTPUT = False
NUM_THREADS = 4
MODEL_NAME = "fu8_large_200hz"
NOISY_NAME = "noiseinput_test.npy"
OUT_NAME = "EEG_denoised_8ch_200hz.npy"


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
    out_path = os.path.join(data_dir, OUT_NAME)

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

    print(f"Ready for inference. Using N = {len(noisy)}, segment_len = {target_len}, channels = {target_ch}")

    total_inference_time = 0.0
    single_times = []
    results = []

    print("\n==== Start inference only ====")
    start_all = time.time()
    for i in range(len(noisy)):
        sample = noisy[i].astype(np.float32).reshape(1, target_len, target_ch)
        interpreter.set_tensor(input_details[0]["index"], sample)
        t0 = time.time()
        interpreter.invoke()
        t1 = time.time()
        dt = t1 - t0
        total_inference_time += dt
        single_times.append(dt)
        out = interpreter.get_tensor(output_details[0]["index"])
        results.append(np.squeeze(out))
    end_all = time.time()

    single_times = np.asarray(single_times, dtype=np.float64)
    decoded = np.asarray(results, dtype=np.float32)

    print("\n==== Inference time (invoke only) ====")
    print(f"Total invoke time: {total_inference_time:.6f} s")
    print(f"Average per segment: {single_times.mean():.6f} s ({single_times.mean() * 1000.0:.2f} ms)")
    print(f"Min / Max per segment: {single_times.min():.6f} s / {single_times.max():.6f} s")
    print(f"Wall-clock time (including loop overhead): {end_all - start_all:.6f} s")

    if SAVE_OUTPUT:
        np.save(out_path, decoded)
        print("Saved denoised output to:", out_path)

    print("Inference only finished.")
