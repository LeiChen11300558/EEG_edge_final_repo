# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import tensorflow as tf
from scipy.signal import resample

# ====================== 1. 路径设置 ======================
current_path = os.path.dirname(os.path.abspath(__file__))

MODEL_STD_NAME = "fu"
MODEL_200_NAME = "fu200hz"

# 定位到上上级的 data 文件夹
data_dir = os.path.abspath(os.path.join(current_path, "..", "..", "data"))

EEG_PATH = os.path.join(data_dir, "EEG_all_epochs.npy")
EOG_PATH = os.path.join(data_dir, "EOG_all_epochs.npy")

print("当前代码路径:", current_path)
print("目标数据路径:", data_dir)


# ====================== 2. 评估指标计算函数 ======================
def rms_value(arr):
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.size == 0: return 0.0
    square = float(np.sum(arr ** 2))
    mean = square / float(arr.size)
    return math.sqrt(mean)


def RRMSE(true, pred):
    num = rms_value(true - pred)
    den = rms_value(true)
    if den == 0.0: return 0.0
    return num / den


def RMSE(true, pred):
    return rms_value(true - pred)


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size == 0: return 0.0
    xm = x - x.mean()
    ym = y - y.mean()
    num = float(np.sum(xm * ym))
    den = math.sqrt(float(np.sum(xm ** 2) * np.sum(ym ** 2)))
    if den == 0.0: return 0.0
    return num / den


def calc_metrics(clean, noisy, decoded):
    """进行零均值化并计算指标"""
    z_clean = clean - np.mean(clean)
    z_noisy = noisy - np.mean(noisy)
    z_decoded = decoded - np.mean(decoded)

    metrics = {
        "noisy_rrmse": RRMSE(z_clean, z_noisy),
        "noisy_cc": pearson_corr(z_clean, z_noisy),
        "deno_rrmse": RRMSE(z_clean, z_decoded),
        "deno_rmse": RMSE(z_clean, z_decoded),
        "deno_cc": pearson_corr(z_clean, z_decoded)
    }
    return metrics


# ====================== 3. 数据加载与合成 ======================
def to_2d(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


print("Loading raw datasets...")
eeg_data = to_2d(np.load(EEG_PATH))
eog_data = to_2d(np.load(EOG_PATH))
N_samples = min(eeg_data.shape[0], eog_data.shape[0])


def create_noisy_signal(clean_eeg, noise_eog, snr_db=0.0):
    rms_eeg = np.sqrt(np.mean(clean_eeg ** 2))
    rms_eog = np.sqrt(np.mean(noise_eog ** 2))
    if rms_eog == 0: return clean_eeg.copy()
    weight = rms_eeg / (10 ** (snr_db / 20.0) * rms_eog)
    return clean_eeg + weight * noise_eog


# ====================== 4. 核心推理函数 ======================
def infer_single_sample(model_name, noisy_1d):
    model_path = os.path.join(current_path, model_name)
    if not os.path.exists(model_path):
        if os.path.exists(model_path + ".tflite"):
            model_path += ".tflite"
        else:
            raise FileNotFoundError(f"找不到模型文件: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    in_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    target_len = in_det['shape'][1]
    orig_len = len(noisy_1d)

    if orig_len != target_len:
        model_input = resample(noisy_1d, target_len)
    else:
        model_input = noisy_1d.copy()

    if len(in_det['shape']) == 3:
        sample_reshaped = model_input.reshape(1, target_len, 1).astype(np.float32)
    else:
        sample_reshaped = model_input.reshape(1, target_len).astype(np.float32)

    interpreter.set_tensor(in_det['index'], sample_reshaped)
    interpreter.invoke()
    out = interpreter.get_tensor(out_det['index'])

    return np.squeeze(out), model_input, target_len


# ====================== 5. 对比画图函数 ======================
def plot_live_comparison(k, snr=0.0):
    if k < 0 or k >= N_samples:
        print("k out of range!")
        return

    clean_k = eeg_data[k]
    noise_k = eog_data[k]
    noisy_k = create_noisy_signal(clean_k, noise_k, snr_db=snr)

    # 运行模型推理
    out_std, noisy_std_in, len_std = infer_single_sample(MODEL_STD_NAME, noisy_k)
    out_200, noisy_200_in, len_200 = infer_single_sample(MODEL_200_NAME, noisy_k)

    FS_ORIGINAL = 256.0
    duration = len(clean_k) / FS_ORIGINAL
    t_std = np.linspace(0, duration, len_std, endpoint=False)
    t_200 = np.linspace(0, duration, len_200, endpoint=False)

    clean_200_ref = resample(clean_k, len_200)

    # 计算指标结果
    m_std = calc_metrics(clean_k, noisy_std_in, out_std)
    m_200 = calc_metrics(clean_200_ref, noisy_200_in, out_200)

    # 准备标题上的文本结果
    title_std = (f"Sample {k} | Standard Model ({MODEL_STD_NAME})\n"
                 f"Input SNR: {snr}dB  |  "
                 f"RRMSE: {m_std['noisy_rrmse']:.3f} -> {m_std['deno_rrmse']:.3f}  |  "
                 f"CC: {m_std['noisy_cc']:.3f} -> {m_std['deno_cc']:.3f}")

    title_200 = (f"Sample {k} | 200Hz Model ({MODEL_200_NAME})\n"
                 f"Input SNR: {snr}dB  |  "
                 f"RRMSE: {m_200['noisy_rrmse']:.3f} -> {m_200['deno_rrmse']:.3f}  |  "
                 f"CC: {m_200['noisy_cc']:.3f} -> {m_200['deno_cc']:.3f}")

    # 开始画图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # 上半部分：标准模型
    ax1.plot(t_std, clean_k, 'k-', label="Clean Target", alpha=0.3)
    ax1.plot(t_std, noisy_std_in, 'r--', label="Noisy Input", alpha=0.4)
    ax1.plot(t_std, out_std, 'b-', label="Denoised Output", linewidth=1.2)
    ax1.set_title(title_std, fontsize=11, fontweight='bold', pad=10)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    # 下半部分：200Hz模型
    ax2.plot(t_200, clean_200_ref, 'k-', label="Clean Target", alpha=0.3)
    ax2.plot(t_200, noisy_200_in, 'r--', label="Noisy Input", alpha=0.4)
    ax2.plot(t_200, out_200, 'g-', label="Denoised Output", linewidth=1.2)
    ax2.set_title(title_200, fontsize=11, fontweight='bold', pad=10)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.xlabel("Time (seconds)")
    fig.text(0.04, 0.5, 'Amplitude (\u03bcV)', va='center', rotation='vertical')
    plt.tight_layout()
    plt.show()


# ====================== 6. 运行入口 ======================
if __name__ == "__main__":
    SAMPLE_INDEX = 12  # <--- 修改查看不同的样本 (多换几个试试)
    TARGET_SNR = -5.0  # <--- SNR越低代表噪声越大

    plot_live_comparison(SAMPLE_INDEX, snr=TARGET_SNR)