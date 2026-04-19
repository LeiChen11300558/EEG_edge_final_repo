import math
import os

import matplotlib.pyplot as plt
import numpy as np


EXPERIMENTS = [
    {
        "name": "FCNN_EOG_bnci2014001_8ch_200hz",
        "nn_output_dir": r"E:\experiment_data\EEG_EEGN\FCNN_EOG_bnci2014001_8ch_200hz\1\nn_output",
        "channel_names": ["FC3", "FC4", "C3", "C4", "CP3", "CP4", "Fz", "Cz"],
        "fs": 200.0,
    },
    # Add another 8-channel experiment here later if you want true model-vs-model comparison.
]

SAMPLE_INDEX = 12


def ensure_3d(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1, 1)
    if arr.ndim == 2:
        return arr[..., np.newaxis]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unsupported array shape: {arr.shape}")


def rms_value(arr):
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    return math.sqrt(float(np.mean(arr ** 2)))


def rrmse(true, pred):
    den = rms_value(true)
    if den == 0.0:
        return 0.0
    return rms_value(true - pred) / den


def rmse(true, pred):
    return rms_value(true - pred)


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size == 0:
        return 0.0
    x = x - np.mean(x)
    y = y - np.mean(y)
    den = math.sqrt(float(np.sum(x ** 2) * np.sum(y ** 2)))
    if den == 0.0:
        return 0.0
    return float(np.sum(x * y)) / den


def calc_metrics(clean, noisy, denoised):
    clean = np.asarray(clean, dtype=np.float64)
    noisy = np.asarray(noisy, dtype=np.float64)
    denoised = np.asarray(denoised, dtype=np.float64)

    clean_z = clean - np.mean(clean)
    noisy_z = noisy - np.mean(noisy)
    deno_z = denoised - np.mean(denoised)

    return {
        "noisy_rrmse": rrmse(clean_z, noisy_z),
        "deno_rrmse": rrmse(clean_z, deno_z),
        "deno_rmse": rmse(clean_z, deno_z),
        "noisy_cc": pearson_corr(clean_z, noisy_z),
        "deno_cc": pearson_corr(clean_z, deno_z),
    }


def calc_dataset_metrics(clean_all, noisy_all, deno_all):
    sample_metrics = []
    for idx in range(clean_all.shape[0]):
        sample_metrics.append(calc_metrics(clean_all[idx], noisy_all[idx], deno_all[idx]))

    keys = sample_metrics[0].keys()
    return {key: float(np.mean([m[key] for m in sample_metrics])) for key in keys}


def load_experiment(exp_cfg):
    base = exp_cfg["nn_output_dir"]
    eeg = ensure_3d(np.load(os.path.join(base, "EEG_test.npy")))
    noisy = ensure_3d(np.load(os.path.join(base, "noiseinput_test.npy")))
    deno = ensure_3d(np.load(os.path.join(base, "Denoiseoutput_test.npy")))
    history_path = os.path.join(base, "loss_history.npy")
    history = np.load(history_path, allow_pickle=True) if os.path.exists(history_path) else None

    if not (eeg.shape == noisy.shape == deno.shape):
        raise ValueError(f"Shape mismatch in {base}: {eeg.shape}, {noisy.shape}, {deno.shape}")

    loaded = dict(exp_cfg)
    loaded["eeg"] = eeg
    loaded["noisy"] = noisy
    loaded["deno"] = deno
    loaded["history"] = history
    loaded["dataset_metrics"] = calc_dataset_metrics(eeg, noisy, deno)
    return loaded


def print_experiment_summary(exp, sample_index):
    eeg = exp["eeg"]
    noisy = exp["noisy"]
    deno = exp["deno"]
    sample_metrics = calc_metrics(eeg[sample_index], noisy[sample_index], deno[sample_index])
    dataset_metrics = exp["dataset_metrics"]

    print(f"\n=== {exp['name']} ===")
    print(f"test shape: {eeg.shape}")
    print(
        "sample metrics:"
        f" noisy_rrmse={sample_metrics['noisy_rrmse']:.4f},"
        f" deno_rrmse={sample_metrics['deno_rrmse']:.4f},"
        f" noisy_cc={sample_metrics['noisy_cc']:.4f},"
        f" deno_cc={sample_metrics['deno_cc']:.4f}"
    )
    print(
        "dataset mean metrics:"
        f" noisy_rrmse={dataset_metrics['noisy_rrmse']:.4f},"
        f" deno_rrmse={dataset_metrics['deno_rrmse']:.4f},"
        f" noisy_cc={dataset_metrics['noisy_cc']:.4f},"
        f" deno_cc={dataset_metrics['deno_cc']:.4f}"
    )


def plot_experiment(exp, sample_index):
    eeg = exp["eeg"]
    noisy = exp["noisy"]
    deno = exp["deno"]
    fs = exp["fs"]
    channel_names = exp["channel_names"]

    if sample_index < 0 or sample_index >= eeg.shape[0]:
        raise IndexError(f"sample_index {sample_index} out of range for {exp['name']}")

    clean_sample = eeg[sample_index]
    noisy_sample = noisy[sample_index]
    deno_sample = deno[sample_index]
    sample_metrics = calc_metrics(clean_sample, noisy_sample, deno_sample)

    n_channels = clean_sample.shape[1]
    times = np.arange(clean_sample.shape[0]) / fs
    ncols = 2
    nrows = int(math.ceil(n_channels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).reshape(-1)

    title = (
        f"{exp['name']} | sample {sample_index}\n"
        f"RRMSE: {sample_metrics['noisy_rrmse']:.4f} -> {sample_metrics['deno_rrmse']:.4f} | "
        f"CC: {sample_metrics['noisy_cc']:.4f} -> {sample_metrics['deno_cc']:.4f}"
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ch in range(n_channels):
        ax = axes[ch]
        label = channel_names[ch] if ch < len(channel_names) else f"Ch {ch + 1}"
        ax.plot(times, clean_sample[:, ch], color="black", alpha=0.45, linewidth=1.0, label="Clean EEG")
        ax.plot(times, noisy_sample[:, ch], color="tab:red", alpha=0.50, linewidth=0.9, linestyle="--", label="Noisy EEG")
        ax.plot(times, deno_sample[:, ch], color="tab:blue", linewidth=1.1, label="Denoised EEG")

        channel_metrics = calc_metrics(clean_sample[:, ch], noisy_sample[:, ch], deno_sample[:, ch])
        ax.set_title(
            f"{label} | RRMSE {channel_metrics['deno_rrmse']:.3f} | CC {channel_metrics['deno_cc']:.3f}",
            fontsize=10,
        )
        ax.grid(True, linestyle=":", alpha=0.5)

    for ax in axes[n_channels:]:
        ax.axis("off")

    axes[0].legend(loc="upper right")
    fig.text(0.5, 0.04, "Time (s)", ha="center")
    fig.text(0.04, 0.5, "Amplitude", va="center", rotation="vertical")
    plt.tight_layout(rect=[0.04, 0.05, 1, 0.93])
    plt.show()


def main():
    loaded_experiments = [load_experiment(exp_cfg) for exp_cfg in EXPERIMENTS]

    print(f"Loaded {len(loaded_experiments)} experiment(s).")
    for exp in loaded_experiments:
        print_experiment_summary(exp, SAMPLE_INDEX)

    for exp in loaded_experiments:
        plot_experiment(exp, SAMPLE_INDEX)


if __name__ == "__main__":
    main()
