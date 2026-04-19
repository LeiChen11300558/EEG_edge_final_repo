import os

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = r"E:\experiment_data\EEG_EEGN\FCNN_EOG_bnci2014001_8ch_200hz\1\nn_output"
CHANNEL_NAMES = ["FC3", "FC4", "C3", "C4", "CP3", "CP4", "Fz", "Cz"]
FS = 200.0


def _load_array(name):
    path = os.path.join(BASE_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)


def _to_3d(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1, 1)
    if arr.ndim == 2:
        return arr[:, :, np.newaxis]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def plot_sample(k, fs=FS):
    noisy = _to_3d(_load_array("noiseinput_test.npy"))
    clean = _to_3d(_load_array("EEG_test.npy"))
    deno = _to_3d(_load_array("Denoiseoutput_test.npy"))

    n_samples = min(noisy.shape[0], clean.shape[0], deno.shape[0])
    if k < 0 or k >= n_samples:
        raise IndexError(f"k out of range: 0 <= k < {n_samples}")

    x_noisy = noisy[k]
    x_clean = clean[k]
    x_deno = deno[k]

    if x_noisy.shape != x_clean.shape or x_noisy.shape != x_deno.shape:
        raise ValueError(f"Shape mismatch: noisy={x_noisy.shape}, clean={x_clean.shape}, deno={x_deno.shape}")

    length, channels = x_noisy.shape
    t = np.arange(length) / fs

    ncols = 2
    nrows = int(np.ceil(channels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.8 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for ch in range(channels):
        ax = axes[ch]
        label = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"Ch {ch + 1}"
        ax.plot(t, x_clean[:, ch], label="Clean EEG", linewidth=1.0, alpha=0.9)
        ax.plot(t, x_noisy[:, ch], label="Noisy EEG", linewidth=0.9, alpha=0.7)
        ax.plot(t, x_deno[:, ch], label="Denoised EEG", linewidth=0.9, alpha=0.9)
        ax.set_title(label)
        ax.grid(True, alpha=0.25)
        if ch % ncols == 0:
            ax.set_ylabel("Amplitude")

    for ax in axes[channels:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(f"Sample {k} - 8-channel Clean vs Noisy vs Denoised", y=0.995)
    fig.supxlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    plot_sample(0)
    input("Press Enter to exit...")
