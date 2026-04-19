# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 13:47:54 2024

@author: m29244lx
"""

# Autoencoder model quantization tests

#%%  import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from path_helper import (
    MODEL_DIR, RUN_MODEL_DIR, X_TEST_NOISY, X_TEST_CLEAN,
    TFLITE_FLOAT16, TFLITE_INT8, TFLITE_REVISION
)
from scipy import signal
import math
import scipy.io
from tensorflow.keras.models import Model

# ---- Config: whether to convert to TFLite on-the-fly in this run
# (suggest False first; enable after downgrading protobuf) ----
DO_CONVERT_TFLITE = True

# Normalize Path -> POSIX string to avoid Windows backslash escaping issues
P = lambda p: p.as_posix()

# ============== SavedModel signature inference (use when it's not a Keras model) ==============
def _load_savedmodel_infer():
    sm = tf.saved_model.load(P(MODEL_DIR))
    infer = sm.signatures.get("serving_default") or next(iter(sm.signatures.values()))
    input_spec_map = infer.structured_input_signature[1]
    input_key = list(input_spec_map.keys())[0]
    input_dtype = input_spec_map[input_key].dtype or tf.float32
    return infer, input_key, input_dtype

def _run_savedmodel(infer, input_key, input_dtype, x):
    out = infer(**{input_key: tf.convert_to_tensor(x, dtype=input_dtype)})
    if isinstance(out, dict):
        y = list(out.values())[0]
    else:
        y = out
    return y.numpy()

#%% load DAE model (keep original usage; only switch to POSIX paths)
autoencoder = tf.keras.models.load_model(P(MODEL_DIR))

# Keep summary; if it's not a Keras model (_UserObject) then gracefully skip
try:
    autoencoder.summary()
    _HAS_ENCODER_DECODER = hasattr(autoencoder, "encoder") and hasattr(autoencoder, "decoder")
except Exception as _e:
    print("[Info] Keras summary unavailable (likely a TF SavedModel _UserObject):", _e)
    _HAS_ENCODER_DECODER = False

#%% Post-training quantization: int8 / float16 conversion
# Guarded by the switch to avoid writing broken tflite when protobuf mismatches
if DO_CONVERT_TFLITE:
    # Dynamic range quantization (int8 weights)
    converter = tf.lite.TFLiteConverter.from_saved_model(P(MODEL_DIR))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model_int8 = converter.convert()
    (RUN_MODEL_DIR / 'autoencoder_int8.tflite').write_bytes(tflite_model_int8)

    # float16 quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(P(MODEL_DIR))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model_float16 = converter.convert()
    (RUN_MODEL_DIR / 'autoencoder_float16.tflite').write_bytes(tflite_model_float16)
else:
    print("[Info] Skip on-the-fly TFLite conversion this run (set DO_CONVERT_TFLITE=True after fixing protobuf).")

#%% Load saved data (keep original logic; only use POSIX paths)
x_test_noisy = np.load(P(X_TEST_NOISY))
x_test_clean = np.load(P(X_TEST_CLEAN))

#%% 1. Run TensorFlow model
#autoencoder = tf.keras.models.load_model(str(MODEL_DIR))
#autoencoder.summary()

if _HAS_ENCODER_DECODER:
    encoded_layer = autoencoder.encoder(x_test_noisy).numpy()
    decoded_layer = autoencoder.decoder(encoded_layer).numpy()
else:
    print("[Info] Fallback to SavedModel signature inference...")
    _infer, _inkey, _indt = _load_savedmodel_infer()
    decoded_layer = _run_savedmodel(_infer, _inkey, _indt, x_test_noisy)

#%% 2. Run TensorFlow Lite models (four-level fallback version)
def runTFLite(input_data, model_path=TFLITE_REVISION):
    """
    Four-level fallback:
      A) Batched + XNNPACK + dynamic resize
      B) Batched + XNNPACK disabled + dynamic resize
      C) Per-sample + XNNPACK + no resize (use default input shape)
      D) Per-sample + XNNPACK disabled + no resize (use default input shape)
    """
    import os

    def _new_interpreter(disable_xnnpack=False):
        if disable_xnnpack:
            os.environ["TF_LITE_DISABLE_XNNPACK"] = "1"
        else:
            if "TF_LITE_DISABLE_XNNPACK" in os.environ:
                del os.environ["TF_LITE_DISABLE_XNNPACK"]
        return tf.lite.Interpreter(model_path=model_path.as_posix())

    # ---------- A/B: batched + resize ----------
    def _try_batched(disable_xnnpack=False):
        interpreter = _new_interpreter(disable_xnnpack)
        input_details = interpreter.get_input_details()
        in_shape = list(input_details[0]["shape"])  # e.g. [1,800] / [-1,800] / [1,800,1]
        desired = in_shape[:]
        if len(desired) >= 1 and desired[0] in (-1, 1):
            desired[0] = int(input_data.shape[0])
        if len(desired) >= 2 and desired[1] != input_data.shape[1]:
            desired[1] = int(input_data.shape[1])
        interpreter.resize_tensor_input(input_details[0]["index"], desired)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        dtype = input_details[0]["dtype"]
        xb = input_data.astype(dtype, copy=False)
        if len(desired) == 3 and desired[2] == 1 and xb.ndim == 2:
            xb = xb[..., np.newaxis]
        interpreter.set_tensor(input_details[0]["index"], xb)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details[0]["index"])
        if y.ndim >= 3 and y.shape[-1] == 1:
            y = np.squeeze(y, axis=-1)
        return y

    # ---------- C/D: per-sample + no resize ----------
    def _try_per_sample(disable_xnnpack=False):
        interpreter = _new_interpreter(disable_xnnpack)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        in_shape = list(input_details[0]["shape"])
        dtype = input_details[0]["dtype"]

        results = []
        for sample in input_data:
            s = sample.astype(dtype, copy=False)
            if len(in_shape) == 2:
                s = s.reshape((1, in_shape[1]))
            elif len(in_shape) == 3 and in_shape[2] == 1:
                s = s.reshape((1, in_shape[1], 1))
            else:
                need = [1] + list(s.shape)
                s = s.reshape(tuple(need))
            interpreter.set_tensor(input_details[0]["index"], s)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]["index"])
            out = np.squeeze(out, axis=0)
            if out.ndim >= 2 and out.shape[-1] == 1:
                out = np.squeeze(out, axis=-1)
            results.append(out)
        return np.stack(results, axis=0)

    # Execute with fallbacks
    try:
        return _try_batched(disable_xnnpack=False)
    except Exception as e:
        print("[TFLite] Batched + XNNPACK failed:", e)
    try:
        return _try_batched(disable_xnnpack=True)
    except Exception as e:
        print("[TFLite] Batched + No-XNNPACK failed:", e)
    try:
        return _try_per_sample(disable_xnnpack=False)
    except Exception as e:
        print("[TFLite] Per-sample + XNNPACK failed:", e)
    try:
        return _try_per_sample(disable_xnnpack=True)
    except Exception as e:
        print("[TFLite] Per-sample + No-XNNPACK failed:", e)
        raise

# Keep TFLite results if you want to use them; if you prefer to use the above SavedModel
# results only, just comment out the next line.
decoded_layer = runTFLite(x_test_noisy, model_path=TFLITE_REVISION)

#%%  Zero centered ####
z_test_noisy = np.zeros(x_test_noisy.shape)
z_test_clean = np.zeros(x_test_clean.shape)
z_decoded_layer = np.zeros(x_test_clean.shape)

for i in range(len(x_test_clean)):
    z_test_noisy[i] = x_test_noisy[i]-np.mean(x_test_noisy[i])
    z_test_clean[i] = x_test_clean[i]-np.mean(x_test_clean[i])
    z_decoded_layer[i] = decoded_layer[i].flatten()-np.mean(decoded_layer[i].flatten())

#%% Detect clean inputs
clean_detect, noisy_detect = [], []
CC_detectClean = np.zeros(shape=(len(z_test_clean),1))
for i in range(len(z_test_clean)):
    CC_detectClean[i] = np.corrcoef(z_test_clean[i], z_test_noisy[i])[0,1]
    if CC_detectClean[i]>0.95:
        clean_detect.append(i)
    else:
        noisy_detect.append(i)

clean_inputs, clean_outputs = [], []
noisy_inputs_EOG, noisy_outputs_EOG, ground_truth_EOG = [], [], []
noisy_inputs_Motion, noisy_outputs_Motion, ground_truth_Motion = [], [], []
noisy_inputs_EMG, noisy_outputs_EMG, ground_truth_EMG = [], [], []

for i in range(len(clean_detect)):
    clean_inputs.append(z_test_noisy[clean_detect[i]])
    clean_outputs.append(z_decoded_layer[clean_detect[i]])

for i in range(len(noisy_detect)):
    if noisy_detect[i]<345:
        noisy_inputs_EOG.append(z_test_noisy[noisy_detect[i]])
        noisy_outputs_EOG.append(z_decoded_layer[noisy_detect[i]])
        ground_truth_EOG.append(z_test_clean[noisy_detect[i]])
    elif noisy_detect[i]>=345 and noisy_detect[i]<967:
        noisy_inputs_Motion.append(z_test_noisy[noisy_detect[i]])
        noisy_outputs_Motion.append(z_decoded_layer[noisy_detect[i]])
        ground_truth_Motion.append(z_test_clean[noisy_detect[i]])
    elif noisy_detect[i]>=967:
        noisy_inputs_EMG.append(z_test_noisy[noisy_detect[i]])
        noisy_outputs_EMG.append(z_decoded_layer[noisy_detect[i]])
        ground_truth_EMG.append(z_test_clean[noisy_detect[i]])

#%% formular define
def rmsValue(arr):
    return math.sqrt(np.mean(np.square(arr)))

def RRMSE(true, pred):
    num = rmsValue(true-pred)
    den = rmsValue(true)
    return num/(den+1e-12)

def RMSE(true, pred):
    return rmsValue(true-pred)

#%% Evaluation
clean_inputs_RRMSE, clean_inputs_RRMSEABS = [], []
for i in range(len(clean_inputs)):
    clean_inputs_RRMSE.append(RRMSE(clean_inputs[i], clean_outputs[i]))
    clean_inputs_RRMSEABS.append(RMSE(clean_inputs[i], clean_outputs[i]))

nperseg, nfft = 200, 800
PSD_len = nfft//2+1
clean_inputs_PSD  = np.zeros((len(clean_inputs), PSD_len))
clean_outputs_PSD = np.zeros((len(clean_inputs), PSD_len))
for i in range(len(clean_inputs)):
    _, pxx = signal.welch(clean_inputs[i],  fs=200, nperseg=nperseg, nfft=nfft); clean_inputs_PSD[i]  = pxx
    _, pxx = signal.welch(clean_outputs[i], fs=200, nperseg=nperseg, nfft=nfft); clean_outputs_PSD[i] = pxx

clean_inputs_PSD_RRMSE, clean_inputs_PSD_RRMSEABS = [], []
for i in range(len(clean_inputs)):
    clean_inputs_PSD_RRMSE.append(RRMSE(clean_inputs_PSD[i], clean_outputs_PSD[i]))
    clean_inputs_PSD_RRMSEABS.append(RMSE(clean_inputs_PSD[i], clean_outputs_PSD[i]))

import scipy.stats
clean_inputs_CC = []
for i in range(len(clean_inputs)):
    clean_inputs_CC.append(scipy.stats.pearsonr(clean_inputs[i], clean_outputs[i]).statistic)

EOG_RRMSE, EOG_RRMSEABS = [], []
for i in range(len(noisy_inputs_EOG)):
    EOG_RRMSE.append(RRMSE(ground_truth_EOG[i], noisy_outputs_EOG[i]))
    EOG_RRMSEABS.append(RMSE(ground_truth_EOG[i], noisy_outputs_EOG[i]))

ground_truth_EOG_PSD  = np.zeros((len(noisy_inputs_EOG), PSD_len))
noisy_outputs_EOG_PSD = np.zeros((len(noisy_inputs_EOG), PSD_len))
for i in range(len(noisy_inputs_EOG)):
    _, pxx = signal.welch(ground_truth_EOG[i], fs=200, nperseg=nperseg, nfft=nfft); ground_truth_EOG_PSD[i]  = pxx
    _, pxx = signal.welch(noisy_outputs_EOG[i], fs=200, nperseg=nperseg, nfft=nfft); noisy_outputs_EOG_PSD[i] = pxx

EOG_PSD_RRMSE, EOG_PSD_RRMSEABS, EOG_CC = [], [], []
for i in range(len(noisy_inputs_EOG)):
    EOG_PSD_RRMSE.append(RRMSE(ground_truth_EOG_PSD[i], noisy_outputs_EOG_PSD[i]))
    EOG_PSD_RRMSEABS.append(RMSE(ground_truth_EOG_PSD[i], noisy_outputs_EOG_PSD[i]))
    EOG_CC.append(scipy.stats.pearsonr(ground_truth_EOG[i], noisy_outputs_EOG[i]).statistic)

Motion_RRMSE, Motion_RRMSEABS = [], []
for i in range(len(noisy_inputs_Motion)):
    Motion_RRMSE.append(RRMSE(ground_truth_Motion[i], noisy_outputs_Motion[i]))
    Motion_RRMSEABS.append(RMSE(ground_truth_Motion[i], noisy_outputs_Motion[i]))

ground_truth_Motion_PSD  = np.zeros((len(noisy_inputs_Motion), PSD_len))
noisy_outputs_Motion_PSD = np.zeros((len(noisy_inputs_Motion), PSD_len))
for i in range(len(noisy_inputs_Motion)):
    _, pxx = signal.welch(ground_truth_Motion[i], fs=200, nperseg=nperseg, nfft=nfft); ground_truth_Motion_PSD[i]  = pxx
    _, pxx = signal.welch(noisy_outputs_Motion[i], fs=200, nperseg=nperseg, nfft=nfft); noisy_outputs_Motion_PSD[i] = pxx

Motion_PSD_RRMSE, Motion_PSD_RRMSEABS, Motion_CC = [], [], []
for i in range(len(noisy_inputs_Motion)):
    Motion_PSD_RRMSE.append(RRMSE(ground_truth_Motion_PSD[i], noisy_outputs_Motion_PSD[i]))
    Motion_PSD_RRMSEABS.append(RMSE(ground_truth_Motion_PSD[i], noisy_outputs_Motion_PSD[i]))
    Motion_CC.append(scipy.stats.pearsonr(ground_truth_Motion[i], noisy_outputs_Motion[i]).statistic)

EMG_RRMSE, EMG_RRMSEABS = [], []
for i in range(len(noisy_inputs_EMG)):
    EMG_RRMSE.append(RRMSE(ground_truth_EMG[i], noisy_outputs_EMG[i]))
    EMG_RRMSEABS.append(RMSE(ground_truth_EMG[i], noisy_outputs_EMG[i]))

ground_truth_EMG_PSD  = np.zeros((len(noisy_inputs_EMG), PSD_len))
noisy_outputs_EMG_PSD = np.zeros((len(noisy_inputs_EMG), PSD_len))
for i in range(len(noisy_inputs_EMG)):
    _, pxx = signal.welch(ground_truth_EMG[i], fs=200, nperseg=nperseg, nfft=nfft); ground_truth_EMG_PSD[i]  = pxx
    _, pxx = signal.welch(noisy_outputs_EMG[i], fs=200, nperseg=nperseg, nfft=nfft); noisy_outputs_EMG_PSD[i] = pxx

EMG_PSD_RRMSE, EMG_PSD_RRMSEABS, EMG_CC = [], [], []
for i in range(len(noisy_inputs_EMG)):
    EMG_PSD_RRMSE.append(RRMSE(ground_truth_EMG_PSD[i], noisy_outputs_EMG_PSD[i]))
    EMG_PSD_RRMSEABS.append(RMSE(ground_truth_EMG_PSD[i], noisy_outputs_EMG_PSD[i]))
    EMG_CC.append(scipy.stats.pearsonr(ground_truth_EMG[i], noisy_outputs_EMG[i]).statistic)

# to numpy
clean_inputs_RRMSE      = np.array(clean_inputs_RRMSE)
clean_inputs_PSD_RRMSE  = np.array(clean_inputs_PSD_RRMSE)
clean_inputs_CC         = np.array(clean_inputs_CC)
clean_inputs_RRMSEABS   = np.array(clean_inputs_RRMSEABS)
clean_inputs_PSD_RRMSEABS = np.array(clean_inputs_PSD_RRMSEABS)

EOG_RRMSE      = np.array(EOG_RRMSE)
EOG_PSD_RRMSE  = np.array(EOG_PSD_RRMSE)
EOG_CC         = np.array(EOG_CC)
EOG_RRMSEABS   = np.array(EOG_RRMSEABS)
EOG_PSD_RRMSEABS = np.array(EOG_PSD_RRMSEABS)

Motion_RRMSE      = np.array(Motion_RRMSE)
Motion_PSD_RRMSE  = np.array(Motion_PSD_RRMSE)
Motion_CC         = np.array(Motion_CC)
Motion_RRMSEABS   = np.array(Motion_RRMSEABS)
Motion_PSD_RRMSEABS = np.array(Motion_PSD_RRMSEABS)

EMG_RRMSE      = np.array(EMG_RRMSE)
EMG_PSD_RRMSE  = np.array(EMG_PSD_RRMSE)
EMG_CC         = np.array(EMG_CC)
EMG_RRMSEABS   = np.array(EMG_RRMSEABS)
EMG_PSD_RRMSEABS = np.array(EMG_PSD_RRMSEABS)

### Print results
print("\n EEG clean input results: ")
print("RRMSE-Time: mean= ", "%.4f" % np.mean(clean_inputs_RRMSE), " ,std= ", "%.4f" % np.std(clean_inputs_RRMSE))
print("RRMSE-Freq: mean= ", "%.4f" % np.mean(clean_inputs_PSD_RRMSE), " ,std= ", "%.4f" % np.std(clean_inputs_PSD_RRMSE))
print("CC: mean= ", "%.4f" % np.mean(clean_inputs_CC), " ,std= ", "%.4f" % np.std(clean_inputs_CC))

print("\n EEG EOG artifacts results:")
print("RRMSE-Time: mean= ","%.4f" % np.mean(EOG_RRMSE), " ,std= ","%.4f" % np.std(EOG_RRMSE))
print("RRMSE-Freq: mean= ","%.4f" % np.mean(EOG_PSD_RRMSE), " ,std= ","%.4f" % np.std(EOG_PSD_RRMSE))
print("CC: mean= ","%.4f" % np.mean(EOG_CC), " ,std= ","%.4f" % np.std(EOG_CC))

print(" \n EEG motion artifacts results:")
print("RRMSE-Time:  mean= ","%.4f" % np.mean(Motion_RRMSE), " ,std= ","%.4f" % np.std(Motion_RRMSE))
print("RRMSE-Freq:  mean= ","%.4f" % np.mean(Motion_PSD_RRMSE), " ,std= ","%.4f" % np.std(Motion_PSD_RRMSE))
print("CC:  mean= ","%.4f" % np.mean(Motion_CC), " ,std= ","%.4f" % np.std(Motion_CC))

print(" \n EEG EMG artifacts results:")
print("RRMSE-Time:  mean= ","%.4f" % np.mean(EMG_RRMSE), " ,std= ","%.4f" % np.std(EMG_RRMSE))
print("RRMSE-Freq:  mean= ","%.4f" % np.mean(EMG_PSD_RRMSE), " ,std= ","%.4f" % np.std(EMG_PSD_RRMSE))
print("CC:  mean= ", "%.4f" % np.mean(EMG_CC), " ,std= ","%.4f" % np.std(EMG_CC))
