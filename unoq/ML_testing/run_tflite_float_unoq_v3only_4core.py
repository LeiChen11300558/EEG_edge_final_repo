# -*- coding: utf-8 -*-

import numpy as np
import os
import time
import tflite_runtime.interpreter as tflite

NUM_THREADS = 4

# ====================== Paths / Config ======================
current_path = os.getcwd()

# TFLite model folder or file name (keep same as your setup)
trained_model_float = "daefloat"  # e.g. "daefloat" or "model.tflite"

# Data files
noisy_data_name = "x_test_noisy1.npy"
clean_data_name = "x_test_clean1.npy"   # optional: only loaded for shape check (can remove)

# Print decoded output matrix (can be huge)
PRINT_DECODED_LAYER = True

# ====================== Load data ======================
full_path_model_float = os.path.join(current_path, trained_model_float)
full_path_noisy = os.path.join(current_path, noisy_data_name)
full_path_clean = os.path.join(current_path, clean_data_name)

x_test_noisy = np.load(full_path_noisy)
# clean not used for metrics anymore; load only if you still want to confirm shapes
x_test_clean = np.load(full_path_clean)

# ensure float32 for TFLite
test_data = x_test_noisy.astype(np.float32)

# if your data is (N, 800) it's fine; if it's (N, 1, 800) etc, flatten per segment
# We'll reshape each segment to (1, 800) at inference.

# ====================== Load TFLite interpreter ======================
print(f"Creating TFLite interpreter with num_threads = {NUM_THREADS}")
interpreter = tflite.Interpreter(model_path=full_path_model_float, num_threads=NUM_THREADS)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ====================== Inference + per-segment time ======================
single_segment_inference_times = []
results = []

start_time = time.time()

for segment in test_data:
    # Make sure each segment is 1D length 800
    segment_1d = np.asarray(segment, dtype=np.float32).reshape(-1)

    # If your model expects exactly 800
    # (optional safety check; you can remove if sure)
    if segment_1d.size != 800:
        raise ValueError(f"Each segment must have 800 values, got {segment_1d.size}")

    # Set input tensor: shape (1, 800)
    interpreter.set_tensor(input_details[0]['index'], segment_1d.reshape(1, 800))

    # Measure single-segment inference time
    t0 = time.time()
    interpreter.invoke()
    t1 = time.time()
    single_segment_inference_times.append(t1 - t0)

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results.append(output_data)

end_time = time.time()

# ====================== Post-process outputs ======================
results = np.array(results)
results = np.squeeze(results)  # remove size-1 dims

# reshape to (N, 800) if possible
decoded_layer = results.reshape(results.shape[0], 800)

# ====================== Print timing ======================
single_segment_inference_times = np.array(single_segment_inference_times, dtype=np.float64)

avg_single = float(np.mean(single_segment_inference_times))
min_single = float(np.min(single_segment_inference_times))
max_single = float(np.max(single_segment_inference_times))
first_single = float(single_segment_inference_times[0]) if single_segment_inference_times.size > 0 else 0.0

print("Total inference time: {:.2f} seconds".format(float(np.sum(single_segment_inference_times))))
print("Average inference time per segment: {:.4f} s ({:.2f} ms)".format(avg_single, avg_single * 1000.0))
print("Min / Max per-segment time: {:.4f} s / {:.4f} s".format(min_single, max_single))
print("First segment inference time: {:.4f} s ({:.2f} ms)".format(first_single, first_single * 1000.0))

# Optional: print decoded outputs
if PRINT_DECODED_LAYER:
    print(decoded_layer)
