# -*- coding: utf-8 -*-

# ====================== import libraries（尽量少库） ======================
import numpy as np
import os
import time
import math

import tflite_runtime.interpreter as tflite

NUM_THREADS = 4


# orgin：
#   import tensorflow as tf
#   import matplotlib.pyplot as plt
#   import pandas as pd
#   from scipy import signal
#   import scipy.io
#   import scipy.stats


# get the current working directory
current_path = os.getcwd()

# define the TFLite model name
trained_model_float = "daefloat"

# define the data files names
noisy_data_name = "x_test_noisy1.npy"
clean_data_name = "x_test_clean1.npy"

# create full paths to the files
full_path_model_float = os.path.join(current_path, trained_model_float)
full_path_noisy = os.path.join(current_path, noisy_data_name)
full_path_clean = os.path.join(current_path, clean_data_name)

# load the data
x_test_noisy = np.load(full_path_noisy)
x_test_clean = np.load(full_path_clean)

# load the TFLite model and allocate tensors
print(f"Creating TFLite interpreter with num_threads = {NUM_THREADS}")
interpreter = tflite.Interpreter(model_path=full_path_model_float, num_threads=NUM_THREADS)
interpreter.allocate_tensors()

# get the input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# prepare the test dataset
test_data = x_test_noisy.astype(np.float32)

# ========= 新增：单个样本推理时间统计 =========
single_segment_inference_times = []

start_time = time.time()

# Run inference on each test segment
results = []

for segment in test_data:
    segment_start = time.time()

    # set input tensor
    interpreter.set_tensor(input_details[0]['index'], segment.reshape((1, 800)))

    # run inference（测量单个样本耗时）
    interpreter.invoke()

    # get the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results.append(output_data)

    segment_end = time.time()
    single_segment_inference_times.append(segment_end - segment_start)

# calculate the elapsed time
end_time = time.time()
inference_time = end_time - start_time

# convert the results to a numPy array
results = np.array(results)
results = np.squeeze(results, axis=(1, 3))

decoded_layer = results

# ========= 打印总推理时间 + 单样本时间 =========
print("Total inference time: {:.2f} seconds".format(inference_time))

single_segment_inference_times = np.array(single_segment_inference_times, dtype=np.float64)
avg_single = np.mean(single_segment_inference_times)
min_single = np.min(single_segment_inference_times)
max_single = np.max(single_segment_inference_times)

print("Average inference time per segment: {:.4f} s ({:.2f} ms)".format(avg_single, avg_single * 1000.0))
print("Min / Max per-segment time: {:.4f} s / {:.4f} s".format(min_single, max_single))

# 也可以顺便看一下第一个样本的大致耗时（可选）
print("First segment inference time: {:.4f} s ({:.2f} ms)".format(
    single_segment_inference_times[0], single_segment_inference_times[0] * 1000.0
))

# 如果还想看推理结果矩阵，可以保留：
print(decoded_layer)

# ====================== Zero centered ======================
# data is in range[0,1], now need to make it zero centered for statistics

z_test_noisy = np.zeros(x_test_noisy.shape)
z_test_clean = np.zeros(x_test_clean.shape)
z_decoded_layer = np.zeros(x_test_clean.shape)

### Zero_centering
for i in range(len(x_test_clean)):
    # noisy set
    z_test_noisy[i] = x_test_noisy[i] - np.mean(x_test_noisy[i])
    # ground truth set
    z_test_clean[i] = x_test_clean[i] - np.mean(x_test_clean[i])
    # denoised set
    z_decoded_layer[i] = decoded_layer[i].flatten() - np.mean(decoded_layer[i].flatten())

# %% Detect clean inputs
clean_detect = []
noisy_detect = []

CC_detectClean = np.zeros(shape=(len(z_test_clean), 1))
for i in range(len(z_test_clean)):
    # calculate cc between noisy and clean version, if they are similar, it means the noisy input is clean
    # if input test_clean and test_noisy is quite different, it means the signal is noisy
    CC_detectClean[i] = np.corrcoef(z_test_clean[i], z_test_noisy[i])[0, 1]
    if CC_detectClean[i] > 0.95:
        clean_detect.append(i)
    else:
        noisy_detect.append(i)

### initialize the lists to store the separated data
clean_inputs = []
clean_outputs = []

noisy_inputs_EOG = []
noisy_inputs_Motion = []
noisy_inputs_EMG = []
noisy_outputs_EOG = []
noisy_outputs_Motion = []
noisy_outputs_EMG = []
ground_truth_EOG = []
ground_truth_Motion = []
ground_truth_EMG = []

for i in range(len(clean_detect)):
    clean_inputs.append(z_test_noisy[clean_detect[i]])
    clean_outputs.append(z_decoded_layer[clean_detect[i]])

for i in range(len(noisy_detect)):
    if noisy_detect[i] < 345:
        noisy_inputs_EOG.append(z_test_noisy[noisy_detect[i]])
        noisy_outputs_EOG.append(z_decoded_layer[noisy_detect[i]])
        ground_truth_EOG.append(z_test_clean[noisy_detect[i]])
    elif noisy_detect[i] >= 345 and noisy_detect[i] < 967:
        noisy_inputs_Motion.append(z_test_noisy[noisy_detect[i]])
        noisy_outputs_Motion.append(z_decoded_layer[noisy_detect[i]])
        ground_truth_Motion.append(z_test_clean[noisy_detect[i]])
    elif noisy_detect[i] >= 967:
        noisy_inputs_EMG.append(z_test_noisy[noisy_detect[i]])
        noisy_outputs_EMG.append(z_decoded_layer[noisy_detect[i]])
        ground_truth_EMG.append(z_test_clean[noisy_detect[i]])


# %% formular define
############### RRMSE (Relative Root Mean Square Error) ##############
# and add pearson_corr and welch_psd in the same place, replacing scipy.stats and scipy.signal.welch.

# Function that Calculate Root Mean Square
def rmsValue(arr):
    square = 0
    mean = 0.0
    root = 0.0
    n = len(arr)
    # Calculate square
    for i in range(0, n):
        square += (arr[i] ** 2)
    # Calculate Mean
    mean = (square / float(n))
    # Calculate Root
    root = math.sqrt(mean)
    return root


def RRMSE(true, pred):
    ### method 1
    # num = np.sum(np.square(true - pred))
    # den = np.sum(np.square(true))
    # squared_error = num/den
    # rrmse_loss = np.sqrt(squared_error)

    ### method 2
    num = rmsValue(true - pred)
    den = rmsValue(true)
    if den == 0:
        return 0.0
    rrmse_loss = num / den
    return rrmse_loss


# calcualte RMSE (Root Mean Square Error)
def RMSE(true, pred):
    return rmsValue(true - pred)


def pearson_corr(x, y):
    """
    alternative：scipy.stats.pearsonr(x, y).statistic
    """
    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()
    if x.size != y.size or x.size == 0:
        return 0.0
    xm = x - x.mean()
    ym = y - y.mean()
    num = float(np.sum(xm * ym))
    den = math.sqrt(float(np.sum(xm ** 2) * np.sum(ym ** 2)))
    if den == 0.0:
        return 0.0
    return num / den


def welch_psd(signal_1d, fs=200.0, nperseg=200, nfft=800):
    """
    alternative：scipy.signal.welch
    """
    x = np.asarray(signal_1d, dtype=np.float32).ravel()
    if x.size == 0:
        freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
        return freqs, np.zeros_like(freqs, dtype=np.float32)

    if nperseg > x.size:
        nperseg = x.size
    step = max(1, nperseg // 2)  # 50% overlap

    window = np.hanning(nperseg).astype(np.float32)
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


# %% Evaluation --- this metrics results are used in final papers --- Just run
###### Clean signal --- Reconstruction
## 1. RRMSE: time domain
clean_inputs_RRMSE = []
clean_inputs_RRMSEABS = []
for i in range(len(clean_inputs)):
    clean_inputs_RRMSE.append(RRMSE(clean_inputs[i], clean_outputs[i]))
    clean_inputs_RRMSEABS.append(RMSE(clean_inputs[i], clean_outputs[i]))

## 2. RRMSE: frequency domain
clean_inputs_PSD_RRMSE = []
clean_inputs_PSD_RRMSEABS = []

nperseg = 200
nfft = 800
PSD_len = nfft // 2 + 1

clean_inputs_PSD = np.zeros(shape=(len(clean_inputs), PSD_len))
clean_outputs_PSD = np.zeros(shape=(len(clean_inputs), PSD_len))

for i in range(len(clean_inputs)):
    f, pxx = welch_psd(clean_inputs[i], fs=200, nperseg=nperseg, nfft=nfft)
    clean_inputs_PSD[i] = pxx

    f, pxx = welch_psd(clean_outputs[i], fs=200, nperseg=nperseg, nfft=nfft)
    clean_outputs_PSD[i] = pxx

for i in range(len(clean_inputs)):
    clean_inputs_PSD_RRMSE.append(RRMSE(clean_inputs_PSD[i], clean_outputs_PSD[i]))
    clean_inputs_PSD_RRMSEABS.append(RMSE(clean_inputs_PSD[i], clean_outputs_PSD[i]))

## 3. CC
clean_inputs_CC = []
for i in range(len(clean_inputs)):
    result = pearson_corr(clean_inputs[i], clean_outputs[i])
    clean_inputs_CC.append(result)

###### EEG/EOG artifacts --- Denoising
## 1. RRMSE: time domain
EOG_RRMSE = []
EOG_RRMSEABS = []
for i in range(len(noisy_inputs_EOG)):
    EOG_RRMSE.append(RRMSE(ground_truth_EOG[i], noisy_outputs_EOG[i]))
    EOG_RRMSEABS.append(RMSE(ground_truth_EOG[i], noisy_outputs_EOG[i]))

## 2. RRMSE: Frequency domain
ground_truth_EOG_PSD = np.zeros(shape=(len(noisy_inputs_EOG), PSD_len))
noisy_outputs_EOG_PSD = np.zeros(shape=(len(noisy_inputs_EOG), PSD_len))

for i in range(len(noisy_inputs_EOG)):
    f, pxx = welch_psd(ground_truth_EOG[i], fs=200, nperseg=nperseg, nfft=nfft)
    ground_truth_EOG_PSD[i] = pxx

    f, pxx = welch_psd(noisy_outputs_EOG[i], fs=200, nperseg=nperseg, nfft=nfft)
    noisy_outputs_EOG_PSD[i] = pxx

EOG_PSD_RRMSE = []
EOG_PSD_RRMSEABS = []
for i in range(len(noisy_inputs_EOG)):
    EOG_PSD_RRMSE.append(RRMSE(ground_truth_EOG_PSD[i], noisy_outputs_EOG_PSD[i]))
    EOG_PSD_RRMSEABS.append(RMSE(ground_truth_EOG_PSD[i], noisy_outputs_EOG_PSD[i]))
## 3. CC
EOG_CC = []
for i in range(len(noisy_inputs_EOG)):
    result = pearson_corr(ground_truth_EOG[i], noisy_outputs_EOG[i])
    EOG_CC.append(result)

###### EEG/Motion artifacts --- Denoising
## 1. RRMSE: time domain
Motion_RRMSE = []
Motion_RRMSEABS = []
for i in range(len(noisy_inputs_Motion)):
    Motion_RRMSE.append(RRMSE(ground_truth_Motion[i], noisy_outputs_Motion[i]))
    Motion_RRMSEABS.append(RMSE(ground_truth_Motion[i], noisy_outputs_Motion[i]))

## 2. RRMSE: Frequency domain
ground_truth_Motion_PSD = np.zeros(shape=(len(noisy_inputs_Motion), PSD_len))
noisy_outputs_Motion_PSD = np.zeros(shape=(len(noisy_inputs_Motion), PSD_len))

for i in range(len(noisy_inputs_Motion)):
    f, pxx = welch_psd(ground_truth_Motion[i], fs=200, nperseg=nperseg, nfft=nfft)
    ground_truth_Motion_PSD[i] = pxx

    f, pxx = welch_psd(noisy_outputs_Motion[i], fs=200, nperseg=nperseg, nfft=nfft)
    noisy_outputs_Motion_PSD[i] = pxx

Motion_PSD_RRMSE = []
Motion_PSD_RRMSEABS = []
for i in range(len(noisy_inputs_Motion)):
    Motion_PSD_RRMSE.append(RRMSE(ground_truth_Motion_PSD[i], noisy_outputs_Motion_PSD[i]))
    Motion_PSD_RRMSEABS.append(RMSE(ground_truth_Motion_PSD[i], noisy_outputs_Motion_PSD[i]))
## 3. CC
Motion_CC = []
for i in range(len(noisy_inputs_Motion)):
    result = pearson_corr(ground_truth_Motion[i], noisy_outputs_Motion[i])
    Motion_CC.append(result)

###### EEG/EMG artifacts --- Denoising
## 1. RRMSE: time domain
EMG_RRMSE = []
EMG_RRMSEABS = []
for i in range(len(noisy_inputs_EMG)):
    EMG_RRMSE.append(RRMSE(ground_truth_EMG[i], noisy_outputs_EMG[i]))
    EMG_RRMSEABS.append(RMSE(ground_truth_EMG[i], noisy_outputs_EMG[i]))
## 2. RRMSE: Frequency domain
ground_truth_EMG_PSD = np.zeros(shape=(len(noisy_inputs_EMG), PSD_len))
noisy_outputs_EMG_PSD = np.zeros(shape=(len(noisy_inputs_EMG), PSD_len))

for i in range(len(noisy_inputs_EMG)):
    f, pxx = welch_psd(ground_truth_EMG[i], fs=200, nperseg=nperseg, nfft=nfft)
    ground_truth_EMG_PSD[i] = pxx

    f, pxx = welch_psd(noisy_outputs_EMG[i], fs=200, nperseg=nperseg, nfft=nfft)
    noisy_outputs_EMG_PSD[i] = pxx

EMG_PSD_RRMSE = []
EMG_PSD_RRMSEABS = []
for i in range(len(noisy_inputs_EMG)):
    EMG_PSD_RRMSE.append(RRMSE(ground_truth_EMG_PSD[i], noisy_outputs_EMG_PSD[i]))
    EMG_PSD_RRMSEABS.append(RMSE(ground_truth_EMG_PSD[i], noisy_outputs_EMG_PSD[i]))

## 3. CC
EMG_CC = []
for i in range(len(noisy_inputs_EMG)):
    result = pearson_corr(ground_truth_EMG[i], noisy_outputs_EMG[i])
    EMG_CC.append(result)

#  Convert list to numpy array
# Clean signal reconstruction
clean_inputs_RRMSE = np.array(clean_inputs_RRMSE)
clean_inputs_PSD_RRMSE = np.array(clean_inputs_PSD_RRMSE)
clean_inputs_CC = np.array(clean_inputs_CC)
#
clean_inputs_RRMSEABS = np.array(clean_inputs_RRMSEABS)
clean_inputs_PSD_RRMSEABS = np.array(clean_inputs_PSD_RRMSEABS)

# EOG artifacts removal
EOG_RRMSE = np.array(EOG_RRMSE)
EOG_PSD_RRMSE = np.array(EOG_PSD_RRMSE)
EOG_CC = np.array(EOG_CC)
#
EOG_RRMSEABS = np.array(EOG_RRMSEABS)
EOG_PSD_RRMSEABS = np.array(EOG_PSD_RRMSEABS)

# Motion artifacts removal
Motion_RRMSE = np.array(Motion_RRMSE)
Motion_PSD_RRMSE = np.array(Motion_PSD_RRMSE)
Motion_CC = np.array(Motion_CC)
#
Motion_RRMSEABS = np.array(Motion_RRMSEABS)
Motion_PSD_RRMSEABS = np.array(Motion_PSD_RRMSEABS)

# EMG artifacts removal
EMG_RRMSE = np.array(EMG_RRMSE)
EMG_PSD_RRMSE = np.array(EMG_PSD_RRMSE)
EMG_CC = np.array(EMG_CC)
#
EMG_RRMSEABS = np.array(EMG_RRMSEABS)
EMG_PSD_RRMSEABS = np.array(EMG_PSD_RRMSEABS)

### Print results
## Clean Input Signal ##
print("\n EEG clean input results: ")
print("RRMSE-Time: mean= ", "%.4f" % np.mean(clean_inputs_RRMSE), " ,std= ", "%.4f" % np.std(clean_inputs_RRMSE))
print("RRMSE-Freq: mean= ", "%.4f" % np.mean(clean_inputs_PSD_RRMSE), " ,std= ",
      "%.4f" % np.std(clean_inputs_PSD_RRMSE))
print("CC: mean= ", "%.4f" % np.mean(clean_inputs_CC), " ,std= ", "%.4f" % np.std(clean_inputs_CC))

## EOG ##
print("\n EEG EOG artifacts results:")
print("RRMSE-Time: mean= ", "%.4f" % np.mean(EOG_RRMSE), " ,std= ", "%.4f" % np.std(EOG_RRMSE))
print("RRMSE-Freq: mean= ", "%.4f" % np.mean(EOG_PSD_RRMSE), " ,std= ", "%.4f" % np.std(EOG_PSD_RRMSE))
print("CC: mean= ", "%.4f" % np.mean(EOG_CC), " ,std= ", "%.4f" % np.std(EOG_CC))

## MOTION
print(" \n EEG motion artifacts results:")
print("RRMSE-Time:  mean= ", "%.4f" % np.mean(Motion_RRMSE), " ,std= ", "%.4f" % np.std(Motion_RRMSE))
print("RRMSE-Freq:  mean= ", "%.4f" % np.mean(Motion_PSD_RRMSE), " ,std= ", "%.4f" % np.std(Motion_PSD_RRMSE))
print("CC:  mean= ", "%.4f" % np.mean(Motion_CC), " ,std= ", "%.4f" % np.std(Motion_CC))

## EMG
print(" \n EEG EMG artifacts results:")
print("RRMSE-Time:  mean= ", "%.4f" % np.mean(EMG_RRMSE), " ,std= ", "%.4f" % np.std(EMG_RRMSE))
print("RRMSE-Freq:  mean= ", "%.4f" % np.mean(EMG_PSD_RRMSE), " ,std= ", "%.4f" % np.std(EMG_PSD_RRMSE))
print("CC:  mean= ", "%.4f" % np.mean(EMG_CC), " ,std= ", "%.4f" % np.std(EMG_CC))
