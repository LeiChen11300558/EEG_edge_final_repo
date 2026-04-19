import numpy as np
import matplotlib.pyplot as plt
import os

# ------------------ 固定工作目录 ------------------
os.chdir(os.path.dirname(__file__))

# ------------------ 数据路径 ------------------
base_dir = r"E:\experiment_data\EEG_EEGN\FCNN_EOG_200hz_rmsp\1\nn_output"

print("Base dir:", base_dir)
print("Files in base_dir:")
print(os.listdir(base_dir))

noisy_path = os.path.join(base_dir, "noiseinput_test.npy")
clean_path = os.path.join(base_dir, "EEG_test.npy")
deno_path  = os.path.join(base_dir, "Denoiseoutput_test.npy")

print("noisy_path:", noisy_path)
print("clean_path:", clean_path)
print("deno_path :", deno_path)

# ---------- 1. 读取数据 ----------
noisy = np.load(noisy_path)
clean = np.load(clean_path)
deno  = np.load(deno_path)

# ---------- 2. 转成 (N, L) ----------
def to_2d(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr

noisy_2d = to_2d(noisy)
clean_2d = to_2d(clean)
deno_2d  = to_2d(deno)

print("noisy_2d shape:", noisy_2d.shape)
print("clean_2d shape:", clean_2d.shape)
print("deno_2d  shape:", deno_2d.shape)

# ---------- 3. 画第 k 条样本 ----------
def plot_sample(k, fs=200.0):   # 200Hz 对应降采样后的模型
    if k < 0 or k >= noisy_2d.shape[0]:
        print(f"k out of range! (0 <= k < {noisy_2d.shape[0]})")
        return

    x_noisy = noisy_2d[k]
    x_clean = clean_2d[k]
    x_deno  = deno_2d[k]

    L = x_noisy.shape[0]
    t = np.arange(L) / fs

    plt.figure(figsize=(10, 5))
    plt.plot(t, x_clean, label="Clean EEG", linewidth=1.1)
    plt.plot(t, x_noisy, label="Noisy EEG", linewidth=1.0)
    plt.plot(t, x_deno, label="Denoised EEG (FCNN)", linewidth=1.0)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Sample {k} - Clean vs Noisy vs Denoised")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------- 4. 主入口 ----------
if __name__ == "__main__":
    print("Try to plot sample 0...")
    plot_sample(0)      # 想看别的样本改这里
    input("Press Enter to exit...")