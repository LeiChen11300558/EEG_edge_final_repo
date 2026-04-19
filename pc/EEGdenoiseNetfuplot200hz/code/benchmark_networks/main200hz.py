import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import partial
from tqdm import tqdm
from IPython.display import clear_output
from data_prepare import *
from Network_structure import *
from loss_function import *
from train_method import *
from save_method import *
import sys
import os
    #sys.path.append('../')
# --- 添加路径代码开始 ---
# 1. 获取当前 main.py 所在的目录 (即 benchmark_networks)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 获取上一级目录 (即 code 文件夹)
parent_dir = os.path.dirname(current_dir)

# 3. 将 code 文件夹加入到系统路径，这样 Python 就能看到 code 下的所有文件夹了
sys.path.append(parent_dir)
# --- 添加路径代码结束 ---
from Novel_CNN import *


def build_fixed_batch_wrapper(base_model, datanum, fixed_batch_size):
    fixed_input = tf.keras.Input(
        batch_shape=(fixed_batch_size, datanum),
        name=f"input_batch{fixed_batch_size}"
    )
    fixed_output = base_model(fixed_input, training=False)
    return tf.keras.Model(
        inputs=fixed_input,
        outputs=fixed_output,
        name=f"{base_model.name}_batch{fixed_batch_size}"
    )

# EEGdenoiseNet V2
# Author: Haoming Zhang
# Here is the main part of the denoising neurl network, We can adjust all the parameter in the user-defined area.
##################################################### 自定义 user-defined ########################################################

epochs = 50    # training epoch
batch_size  = 40    # training batch size
combin_num = 10    # combin EEG and noise ? times
denoise_network = 'fcNN'    # fcNN & Simple_CNN & Complex_CNN & RNN_lstm  & Novel_CNN
noise_type = 'EOG'

# ============ 采样率设置 ============
# 原始采样率和目标采样率，设置 target_fs = None 则不做降采样
if noise_type == 'EOG':
    orig_fs = 256       # 原始采样率 (Hz)
    target_fs = 200     # 目标采样率 (Hz)，设为 None 则保持原始
elif noise_type == 'EMG':
    orig_fs = 512
    target_fs = None    # EMG 暂不降采样，如需要可改为目标值
# ====================================

result_location = r'E:/experiment_data/EEG_EEGN/'     #  Where to export network results   ############ change it to your own location #########
foldername = 'FCNN_EOG_200hz_rmsp'   # 改名以区分 200Hz 版本
os.environ['CUDA_VISIBLE_DEVICES']='0'
save_train = False
save_vali = False
save_test = True


################################################## optimizer adjust parameter  ####################################################
rmsp=tf.optimizers.RMSprop(lr=0.00005, rho=0.9)
adam=tf.optimizers.Adam(lr=0.00005, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
sgd=tf.keras.optimizers.legacy.SGD(learning_rate=0.0002, momentum=0.9, decay=0.0, nesterov=False)

optimizer = rmsp


# We have reserved an example of importing an existing network
'''
path = os.path.join(result_location, foldername, "denoised_model")
denoiseNN = tf.keras.models.load_model(path)
'''
#################################################### 数据输入 Import data #####################################################

file_location = '../../data/'                   ############ change it to your own location #########
if noise_type == 'EOG':
    EEG_all = np.load( file_location + 'EEG_all_epochs.npy')
    noise_all = np.load( file_location + 'EOG_all_epochs.npy')
elif noise_type == 'EMG':
    EEG_all = np.load( file_location + 'EEG_all_epochs_512hz.npy')
    noise_all = np.load( file_location + 'EMG_all_epochs_512hz.npy')

# ============ 降采样处理 ============
if target_fs is not None and target_fs != orig_fs:
    orig_points = EEG_all.shape[1]
    target_points = int(orig_points * target_fs / orig_fs)  # 512 * 200/256 = 400
    print(f'Resampling: {orig_fs}Hz ({orig_points} points) -> {target_fs}Hz ({target_points} points)')
    EEG_all = resample_data(EEG_all, orig_points, target_points)
    noise_all = resample_data(noise_all, orig_points, target_points)
    print(f'After resampling: EEG shape={EEG_all.shape}, noise shape={noise_all.shape}')
else:
    print(f'No resampling, keeping original {orig_fs}Hz')

# datanum 自动从数据维度获取
datanum = EEG_all.shape[1]
print(f'datanum = {datanum}')
# ====================================

############################################################# Running #############################################################
#for i in range(10):
i = 1     # We run each NN for 10 times to increase  the  statistical  power  of  our  results
noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, test_std_VALUE = prepare_data(
    EEG_all = EEG_all,
    noise_all = noise_all,
    combin_num = 10,
    train_per = 0.8,
    noise_type = noise_type
)


if denoise_network == 'fcNN':
    model = fcNN(datanum)

elif denoise_network == 'Simple_CNN':
    model = simple_CNN(datanum)

elif denoise_network == 'Complex_CNN':
    model = Complex_CNN(datanum)

elif denoise_network == 'RNN_lstm':
    model = RNN_lstm(datanum)

elif denoise_network == 'Novel_CNN':
    model = Novel_CNN(datanum)

else:
    print('NN name arror')


saved_model, history = train(
    model, noiseEEG_train, EEG_train, noiseEEG_val, EEG_val,
    epochs, batch_size, optimizer, denoise_network,
    result_location, foldername , train_num = str(i)
)

#denoised_test, test_mse = test_step(saved_model, noiseEEG_test, EEG_test)

# save signal
save_eeg(
    saved_model, result_location, foldername, save_train, save_vali, save_test,
    noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test,
    train_num = str(i)
)
np.save(
    result_location +'/'+ foldername + '/'+ str(i)  +'/'+ "nn_output" + '/'+ 'loss_history.npy',
    history
)

######################################## 保存 Keras 模型 + 导出 TFLite ########################################

# 1. 先把 Keras 模型保存到一个固定目录（方便以后继续加载）
model_dir = os.path.join(result_location, foldername, str(i), "denoise_model")
os.makedirs(model_dir, exist_ok=True)

keras_model_path = os.path.join(model_dir, "denoise_model.keras")
tf.keras.models.save_model(saved_model, keras_model_path)

# 2. 直接从内存中的 Keras 模型转换为 TFLite（最简单的方式）
converter = tf.lite.TFLiteConverter.from_keras_model(saved_model)

# 如果你后面想做量化，这里可以加一些设置，比如：
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# 3. 把 tflite 模型写入文件
tflite_model_path = os.path.join(model_dir, "denoise_model.tflite")
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("TFLite 模型已保存到:", tflite_model_path)

if denoise_network == 'fcNN':
    batch4_model = build_fixed_batch_wrapper(saved_model, datanum, fixed_batch_size=4)
    batch4_keras_path = os.path.join(model_dir, "denoise_model_batch4.keras")
    tf.keras.models.save_model(batch4_model, batch4_keras_path)

    batch4_converter = tf.lite.TFLiteConverter.from_keras_model(batch4_model)
    batch4_tflite_model = batch4_converter.convert()

    batch4_tflite_path = os.path.join(model_dir, "denoise_model_batch4.tflite")
    with open(batch4_tflite_path, "wb") as f:
        f.write(batch4_tflite_model)

    print("Batch-4 TFLite model saved to:", batch4_tflite_path)
