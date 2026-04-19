import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import clear_output
from tensorflow.keras import datasets, layers, models
from tqdm import tqdm

from data_prepare import *
from Network_structure import *
from loss_function import *
from save_method import *
from train_method import *

# Build import path for Novel_CNN
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from Novel_CNN import *


def build_fixed_batch_wrapper(base_model, datanum, fixed_batch_size):
    fixed_input = tf.keras.Input(
        batch_shape=(fixed_batch_size,) + tuple(base_model.input_shape[1:]),
        name=f"input_batch{fixed_batch_size}",
    )
    fixed_output = base_model(fixed_input, training=False)
    return tf.keras.Model(
        inputs=fixed_input,
        outputs=fixed_output,
        name=f"{base_model.name}_batch{fixed_batch_size}",
    )


# User-defined parameters
epochs = 100
batch_size = 40
combin_num = 10
denoise_network = "fcNN"  # fcNN & Simple_CNN & Complex_CNN & RNN_lstm & Novel_CNN
noise_type = "EOG"
dataset_name = "bnci2014001_8ch"

if noise_type == "EOG":
    orig_fs = 250 if dataset_name.startswith("bnci2014001_8ch") else 256
    target_fs = 200
elif noise_type == "EMG":
    orig_fs = 512
    target_fs = None
else:
    raise ValueError(f"Unsupported noise_type: {noise_type}")

result_location = r"E:/experiment_data/EEG_EEGN/"
if dataset_name == "bnci2014001_8ch":
    foldername = "FCNN_EOG_bnci2014001_8ch_200hz_large_ep100"
elif dataset_name == "bnci2014001_8ch_clean_top40":
    foldername = "FCNN_EOG_bnci2014001_8ch_clean_top40_200hz_ep100"
else:
    foldername = "FCNN_EOG_200hz_rmsp"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
save_train = False
save_vali = False
save_test = True


# Optimizers
rmsp = tf.optimizers.RMSprop(lr=0.00005, rho=0.9)
adam = tf.optimizers.Adam(lr=0.00005, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
sgd = tf.keras.optimizers.legacy.SGD(learning_rate=0.0002, momentum=0.9, decay=0.0, nesterov=False)
optimizer = rmsp


def load_dataset():
    file_location = "../../data/"

    if noise_type == "EOG":
        if dataset_name == "bnci2014001_8ch":
            file_location = "../../data/bnci2014001_8ch_merged/"
            eeg_all = np.load(file_location + "EEG_all_epochs_bnci2014001_8ch.npy")
            noise_all = np.load(file_location + "EOG_projected_epochs_bnci2014001_8ch.npy")
        elif dataset_name == "bnci2014001_8ch_clean_top40":
            file_location = "../../data/bnci2014001_8ch_filtered/"
            eeg_all = np.load(file_location + "EEG_all_epochs_bnci2014001_8ch_clean_top40.npy")
            noise_all = np.load(file_location + "EOG_projected_epochs_bnci2014001_8ch_clean_top40.npy")
        else:
            eeg_all = np.load(file_location + "EEG_all_epochs.npy")
            noise_all = np.load(file_location + "EOG_all_epochs.npy")
    elif noise_type == "EMG":
        eeg_all = np.load(file_location + "EEG_all_epochs_512hz.npy")
        noise_all = np.load(file_location + "EMG_all_epochs_512hz.npy")
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")

    eeg_all = ensure_3d(eeg_all)
    noise_all = ensure_3d(noise_all)
    return eeg_all, noise_all, file_location


EEG_all, noise_all, file_location = load_dataset()

if target_fs is not None and target_fs != orig_fs:
    orig_points = EEG_all.shape[1]
    target_points = int(orig_points * target_fs / orig_fs)
    print(f"Resampling: {orig_fs}Hz ({orig_points} points) -> {target_fs}Hz ({target_points} points)")
    EEG_all = resample_data(EEG_all, orig_points, target_points)
    noise_all = resample_data(noise_all, orig_points, target_points)
    print(f"After resampling: EEG shape={EEG_all.shape}, noise shape={noise_all.shape}")
else:
    print(f"No resampling, keeping original {orig_fs}Hz")

datanum = EEG_all.shape[1]
channelnum = EEG_all.shape[2]
print(f"datanum = {datanum}, channelnum = {channelnum}")
print(f"dataset_name = {dataset_name}")
print(f"file_location = {file_location}")


noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, test_std_VALUE = prepare_data(
    EEG_all=EEG_all,
    noise_all=noise_all,
    combin_num=combin_num,
    train_per=0.8,
    noise_type=noise_type,
)


if denoise_network == "fcNN":
    model = fcNN(datanum, channelnum)
elif denoise_network == "Simple_CNN":
    model = simple_CNN(datanum, channelnum)
elif denoise_network == "Complex_CNN":
    model = Complex_CNN(datanum, channelnum)
elif denoise_network == "RNN_lstm":
    model = RNN_lstm(datanum, channelnum)
elif denoise_network == "Novel_CNN":
    model = Novel_CNN(datanum, channelnum)
else:
    raise ValueError("NN name error")


i = 1
saved_model, history = train(
    model,
    noiseEEG_train,
    EEG_train,
    noiseEEG_val,
    EEG_val,
    epochs,
    batch_size,
    optimizer,
    denoise_network,
    result_location,
    foldername,
    train_num=str(i),
)

save_eeg(
    saved_model,
    result_location,
    foldername,
    save_train,
    save_vali,
    save_test,
    noiseEEG_train,
    EEG_train,
    noiseEEG_val,
    EEG_val,
    noiseEEG_test,
    EEG_test,
    train_num=str(i),
)

np.save(
    result_location + "/" + foldername + "/" + str(i) + "/" + "nn_output" + "/" + "loss_history.npy",
    history,
)


# Save Keras and TFLite models
model_dir = os.path.join(result_location, foldername, str(i), "denoise_model")
os.makedirs(model_dir, exist_ok=True)

keras_model_path = os.path.join(model_dir, "denoise_model_savedmodel")
saved_model.save(keras_model_path, save_format="tf")

converter = tf.lite.TFLiteConverter.from_keras_model(saved_model)
tflite_model = converter.convert()

tflite_model_path = os.path.join(model_dir, "denoise_model.tflite")
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("TFLite model saved to:", tflite_model_path)

if denoise_network == "fcNN":
    batch4_model = build_fixed_batch_wrapper(saved_model, datanum, fixed_batch_size=4)
    batch4_keras_path = os.path.join(model_dir, "denoise_model_batch4_savedmodel")
    batch4_model.save(batch4_keras_path, save_format="tf")

    batch4_converter = tf.lite.TFLiteConverter.from_keras_model(batch4_model)
    batch4_tflite_model = batch4_converter.convert()

    batch4_tflite_path = os.path.join(model_dir, "denoise_model_batch4.tflite")
    with open(batch4_tflite_path, "wb") as f:
        f.write(batch4_tflite_model)

    print("Batch-4 TFLite model saved to:", batch4_tflite_path)
