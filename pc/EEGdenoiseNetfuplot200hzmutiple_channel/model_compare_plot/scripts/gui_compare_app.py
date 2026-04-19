import json
import math
import time
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter as TfInterpreter

    class _TfLiteWrapper:
        Interpreter = TfInterpreter

    tflite = _TfLiteWrapper()


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
CURRENT_LANG = "zh"
NUM_THREADS = 4


MODEL_CONFIGS = {
    "fcnn_std": {
        "family": "fcnn",
        "display_en": "FCNN Standard",
        "display_zh": "??FCNN",
        "model_path": MODEL_DIR / "fcnn" / "fu",
        "noisy_path": DATA_DIR / "fcnn" / "noiseinput_test.npy",
        "clean_path": DATA_DIR / "fcnn" / "EEG_test.npy",
        "default_fs": 256.0,
        "fixed_batch": None,
    },
    "fcnn_200hz": {
        "family": "fcnn",
        "display_en": "FCNN 200Hz",
        "display_zh": "200Hz FCNN",
        "model_path": MODEL_DIR / "fcnn" / "fu200hz",
        "noisy_path": DATA_DIR / "fcnn" / "noiseinput_test.npy",
        "clean_path": DATA_DIR / "fcnn" / "EEG_test.npy",
        "default_fs": 200.0,
        "fixed_batch": None,
    },
    "fcnn_200hz_batch4": {
        "family": "fcnn",
        "display_en": "FCNN 200Hz Batch4",
        "display_zh": "200Hz FCNN Batch4",
        "model_path": MODEL_DIR / "fcnn" / "fu200hzbatch4",
        "noisy_path": DATA_DIR / "fcnn" / "noiseinput_test.npy",
        "clean_path": DATA_DIR / "fcnn" / "EEG_test.npy",
        "default_fs": 200.0,
        "fixed_batch": 4,
    },
    "fcnn_bnci8_large_ep100_200hz": {
        "family": "fcnn",
        "display_en": "FCNN BNCI 8ch Large Ep100 200Hz",
        "display_zh": "FCNN BNCI 8閫氶亾 Large Ep100 200Hz",
        "model_path": MODEL_DIR / "fcnn" / "fu_bnci8_large_ep100_200hz",
        "noisy_path": DATA_DIR / "fcnn_bnci8_large_ep100" / "noiseinput_test.npy",
        "clean_path": DATA_DIR / "fcnn_bnci8_large_ep100" / "EEG_test.npy",
        "precomputed_denoised_path": DATA_DIR / "fcnn_bnci8_large_ep100" / "Denoiseoutput_test.npy",
        "channel_names": ["FC3", "FC4", "C3", "C4", "CP3", "CP4", "Fz", "Cz"],
        "default_fs": 200.0,
        "fixed_batch": None,
    },
    "autoencoder": {
        "family": "autoencoder",
        "display_en": "Autoencoder",
        "display_zh": "????",
        "model_path": MODEL_DIR / "autoencoder" / "daefloat",
        "noisy_path": DATA_DIR / "autoencoder" / "x_test_noisy1.npy",
        "clean_path": DATA_DIR / "autoencoder" / "x_test_clean1.npy",
        "default_fs": 200.0,
        "fixed_batch": None,
    },
    "autoencoder_self": {
        "family": "autoencoder",
        "display_en": "Autoencoder Self",
        "display_zh": "????Self",
        "model_path": MODEL_DIR / "autoencoder" / "daefloatself",
        "noisy_path": DATA_DIR / "autoencoder" / "x_test_noisy1.npy",
        "clean_path": DATA_DIR / "autoencoder" / "x_test_clean1.npy",
        "default_fs": 200.0,
        "fixed_batch": None,
    },
}
METRICS_CACHE = {}


I18N = {
    "en": {
        "title": "EEG Model Compare Tool",
        "language": "Language",
        "mode": "Mode",
        "single_mode": "Single Model",
        "multi_mode": "Multi Model",
        "models": "Models",
        "sample_index": "Sample Index",
        "same_sample": "Use same sample index for all selected models",
        "enable_compare_plot": "Enable comparison plot",
        "show_batch4": "Show all 4 outputs for batch4 model",
        "run": "Run",
        "full_ae_eval": "Run Full Eval",
        "clear": "Clear",
        "status_ready": "Ready.",
        "status_running": "Running...",
        "single_plot_title": "Clean vs Noisy vs Denoised",
        "compare_plot_title": "Multi-Model Comparison",
        "error_select_model": "Please select at least one model.",
        "error_sample": "Sample index is invalid.",
        "error_missing_file": "Missing file",
        "done": "Finished.",
        "save_json": "Save last result to JSON",
        "comparison_note": "Different lengths are aligned automatically before plotting.",
        "ae_note": "Autoencoder metrics use zero-centered logic to match the board script.",
    },
    "zh": {
        "title": "EEG??????",
        "language": "??",
        "mode": "??",
        "single_mode": "?????",
        "multi_mode": "?????",
        "models": "????",
        "sample_index": "????",
        "same_sample": "???????????????",
        "enable_compare_plot": "??????",
        "show_batch4": "?batch4????4????",
        "run": "??",
        "full_ae_eval": "运行全量评估",
        "clear": "??",
        "status_ready": "???",
        "status_running": "???...",
        "single_plot_title": "???? / ???? / ????",
        "compare_plot_title": "??????",
        "error_select_model": "??????????",
        "error_sample": "???????",
        "error_missing_file": "????",
        "done": "???",
        "save_json": "???????JSON",
        "comparison_note": "??????????????",
        "ae_note": "Autoencoder?????????zero-centered?????",
    },
}


def to_2d(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def to_3d(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1, 1)
    if arr.ndim == 2:
        return arr[..., np.newaxis]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unsupported data shape: {arr.shape}")


def resolve_model_path(path: Path) -> Path:
    if path.exists():
        return path
    if path.with_suffix('.tflite').exists():
        return path.with_suffix('.tflite')
    raise FileNotFoundError(str(path))


def resample_1d(signal_1d, target_len):
    signal_1d = np.asarray(signal_1d, dtype=np.float32).reshape(-1)
    if signal_1d.size == target_len:
        return signal_1d
    old_x = np.linspace(0.0, 1.0, signal_1d.size, dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    return np.interp(new_x, old_x, signal_1d).astype(np.float32)


def rms_value(arr):
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    return math.sqrt(float(np.sum(arr ** 2)) / float(arr.size))


def rmse(true, pred):
    return rms_value(np.asarray(true) - np.asarray(pred))


def rrmse(true, pred):
    den = rms_value(true)
    if den == 0.0:
        return 0.0
    return rmse(true, pred) / den


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size == 0:
        return 0.0
    xm = x - x.mean()
    ym = y - y.mean()
    den = math.sqrt(float(np.sum(xm ** 2) * np.sum(ym ** 2)))
    if den == 0.0:
        return 0.0
    return float(np.sum(xm * ym)) / den


def auto_ylim(series_list, family=None):
    arr = np.concatenate([np.asarray(s, dtype=np.float32).reshape(-1) for s in series_list])
    if arr.size == 0:
        return None
    if family == "autoencoder":
        center = float(np.median(arr))
        max_dev = float(np.max(np.abs(arr - center)))
        half = max(max_dev * 3.5, 1e-3)
        return center - half, center + half
    low = float(np.percentile(arr, 2))
    high = float(np.percentile(arr, 98))
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if low == high:
        margin = max(1.0, abs(low) * 0.1 + 1e-3)
        return low - margin, high + margin
    margin = (high - low) * 0.12
    return low - margin, high + margin


def zero_center(x):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x - np.mean(x)


def zero_center_nd(x):
    x = np.asarray(x, dtype=np.float32)
    return x - np.mean(x)


def resample_sample(sample, target_len):
    sample = np.asarray(sample, dtype=np.float32)
    if sample.ndim == 1:
        return resample_1d(sample, target_len)
    channels = [resample_1d(sample[:, ch], target_len) for ch in range(sample.shape[1])]
    return np.stack(channels, axis=-1).astype(np.float32)


def calc_metrics_nd(clean, noisy, denoised):
    clean_z = zero_center_nd(clean)
    noisy_z = zero_center_nd(noisy)
    denoised_z = zero_center_nd(denoised)
    if clean_z.ndim == 1:
        clean_ch = clean_z.reshape(-1, 1)
        noisy_ch = noisy_z.reshape(-1, 1)
        denoised_ch = denoised_z.reshape(-1, 1)
    else:
        clean_ch = clean_z
        noisy_ch = noisy_z
        denoised_ch = denoised_z

    channel_cc = []
    baseline_channel_cc = []
    for ch in range(clean_ch.shape[1]):
        channel_cc.append(float(pearson_corr(clean_ch[:, ch], denoised_ch[:, ch])))
        baseline_channel_cc.append(float(pearson_corr(clean_ch[:, ch], noisy_ch[:, ch])))

    return {
        "rmse": float(rmse(clean_z, denoised_z)),
        "rrmse": float(rrmse(clean_z, denoised_z)),
        "pearson_cc": float(pearson_corr(clean_z, denoised_z)),
        "channel_mean_cc": float(np.mean(channel_cc)),
        "channel_min_cc": float(np.min(channel_cc)),
        "channel_max_cc": float(np.max(channel_cc)),
        "baseline_rrmse": float(rrmse(clean_z, noisy_z)),
        "baseline_pearson_cc": float(pearson_corr(clean_z, noisy_z)),
        "baseline_channel_mean_cc": float(np.mean(baseline_channel_cc)),
        "baseline_channel_min_cc": float(np.min(baseline_channel_cc)),
        "baseline_channel_max_cc": float(np.max(baseline_channel_cc)),
    }


def summarize_precomputed_metrics(config):
    cache_key = str(config.get("precomputed_denoised_path", ""))
    if not cache_key:
        return None
    if cache_key in METRICS_CACHE:
        return METRICS_CACHE[cache_key]

    den_path = config.get("precomputed_denoised_path")
    if den_path is None or not den_path.exists():
        return None

    clean_all = to_3d(np.load(config["clean_path"], mmap_mode="r"))
    noisy_all = to_3d(np.load(config["noisy_path"], mmap_mode="r"))
    denoised_all = to_3d(np.load(den_path, mmap_mode="r"))
    total_n = min(clean_all.shape[0], noisy_all.shape[0], denoised_all.shape[0])

    rrmse_vals = []
    base_rrmse_vals = []
    cc_vals = []
    base_cc_vals = []
    ch_mean_cc_vals = []
    ch_min_cc_vals = []
    ch_max_cc_vals = []
    base_ch_mean_cc_vals = []
    base_ch_min_cc_vals = []
    base_ch_max_cc_vals = []
    for i in range(total_n):
        metrics = calc_metrics_nd(clean_all[i], noisy_all[i], denoised_all[i])
        rrmse_vals.append(metrics["rrmse"])
        base_rrmse_vals.append(metrics["baseline_rrmse"])
        cc_vals.append(metrics["pearson_cc"])
        base_cc_vals.append(metrics["baseline_pearson_cc"])
        ch_mean_cc_vals.append(metrics["channel_mean_cc"])
        ch_min_cc_vals.append(metrics["channel_min_cc"])
        ch_max_cc_vals.append(metrics["channel_max_cc"])
        base_ch_mean_cc_vals.append(metrics["baseline_channel_mean_cc"])
        base_ch_min_cc_vals.append(metrics["baseline_channel_min_cc"])
        base_ch_max_cc_vals.append(metrics["baseline_channel_max_cc"])

    summary = {
        "count": int(total_n),
        "rrmse_mean": float(np.mean(rrmse_vals)),
        "baseline_rrmse_mean": float(np.mean(base_rrmse_vals)),
        "pearson_cc_mean": float(np.mean(cc_vals)),
        "baseline_pearson_cc_mean": float(np.mean(base_cc_vals)),
        "channel_mean_cc_mean": float(np.mean(ch_mean_cc_vals)),
        "channel_min_cc_mean": float(np.mean(ch_min_cc_vals)),
        "channel_max_cc_mean": float(np.mean(ch_max_cc_vals)),
        "baseline_channel_mean_cc_mean": float(np.mean(base_ch_mean_cc_vals)),
        "baseline_channel_min_cc_mean": float(np.mean(base_ch_min_cc_vals)),
        "baseline_channel_max_cc_mean": float(np.mean(base_ch_max_cc_vals)),
    }
    METRICS_CACHE[cache_key] = summary
    return summary


def welch_psd(signal_1d, fs=200.0, nperseg=200, nfft=800):
    x = np.asarray(signal_1d, dtype=np.float32).ravel()
    if x.size == 0:
        freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
        return freqs, np.zeros_like(freqs, dtype=np.float32)
    if nperseg > x.size:
        nperseg = x.size
    step = max(1, nperseg // 2)
    window = np.hanning(nperseg).astype(np.float32)
    scale = fs * float(np.sum(window ** 2))
    seg_psds = []
    for start in range(0, x.size - nperseg + 1, step):
        seg = x[start:start + nperseg] * window
        spec = np.fft.rfft(seg, n=nfft)
        seg_psds.append((np.abs(spec) ** 2) / scale)
    if not seg_psds:
        seg = x[:nperseg] * window
        spec = np.fft.rfft(seg, n=nfft)
        seg_psds = [(np.abs(spec) ** 2) / scale]
    seg_psds = np.stack(seg_psds, axis=0)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    return freqs, np.mean(seg_psds, axis=0)


def autoencoder_sample_type(clean, noisy, sample_index):
    cc_detect = pearson_corr(zero_center(clean), zero_center(noisy))
    if cc_detect > 0.95:
        return "clean", cc_detect
    if sample_index < 345:
        return "EOG", cc_detect
    if sample_index < 967:
        return "Motion", cc_detect
    return "EMG", cc_detect


def summarize_metric(values):
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {"count": int(arr.size), "mean": float(np.mean(arr)), "std": float(np.std(arr))}


def run_inference_for_sample(config, sample_index):
    model_path = resolve_model_path(config["model_path"])
    noisy_all = to_3d(np.load(config["noisy_path"]))
    clean_all = to_3d(np.load(config["clean_path"]))
    total_n = min(len(noisy_all), len(clean_all))
    if sample_index < 0 or sample_index >= total_n:
        raise IndexError(f"sample_index={sample_index}, total_n={total_n}")

    interpreter = tflite.Interpreter(model_path=str(model_path), num_threads=NUM_THREADS)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_shape = input_details["shape"]
    target_len = int(input_shape[1])
    target_ch = int(input_shape[2]) if len(input_shape) == 3 else 1
    fixed_batch = config["fixed_batch"]
    batch_size = fixed_batch or 1
    if sample_index + batch_size > total_n:
        raise IndexError(f"Need {batch_size} consecutive samples starting at {sample_index}, but dataset has {total_n} samples")

    noisy_batch_raw = noisy_all[sample_index:sample_index + batch_size].astype(np.float32)
    clean_batch_raw = clean_all[sample_index:sample_index + batch_size].astype(np.float32)
    noisy_batch = np.stack([resample_sample(x, target_len) for x in noisy_batch_raw], axis=0)
    clean_batch = np.stack([resample_sample(x, target_len) for x in clean_batch_raw], axis=0)

    if noisy_batch.shape[-1] != target_ch:
        raise ValueError(f"Channel mismatch: data has {noisy_batch.shape[-1]} channels, model expects {target_ch}")

    if fixed_batch is not None:
        if len(input_shape) == 2:
            input_tensor = noisy_batch[:, :, 0].astype(np.float32)
        else:
            input_tensor = noisy_batch.astype(np.float32)
    else:
        if len(input_shape) == 3:
            input_tensor = noisy_batch[:1].reshape(1, target_len, target_ch).astype(np.float32)
        else:
            input_tensor = noisy_batch[:1, :, 0].reshape(1, target_len).astype(np.float32)
        clean_batch = clean_batch[:1]
        noisy_batch = noisy_batch[:1]
        batch_size = 1

    interpreter.set_tensor(input_details["index"], input_tensor)
    t0 = time.perf_counter()
    interpreter.invoke()
    t1 = time.perf_counter()
    output = interpreter.get_tensor(output_details["index"])
    denoised_batch = to_3d(output)[:batch_size].astype(np.float32)

    primary_clean = clean_batch[0]
    primary_noisy = noisy_batch[0]
    primary_denoised = denoised_batch[0]
    total_inference_seconds = float(t1 - t0)

    metric_clean = primary_clean
    metric_noisy = primary_noisy
    metric_denoised = primary_denoised
    sample_type = None
    clean_detect_cc = None
    if config["family"] == "autoencoder":
        metric_clean = zero_center(primary_clean)
        metric_noisy = zero_center(primary_noisy)
        metric_denoised = zero_center(primary_denoised)
        sample_type, clean_detect_cc = autoencoder_sample_type(primary_clean, primary_noisy, sample_index)

    metrics = {
        "length": int(target_len),
        "channels": int(primary_clean.shape[-1]),
        "invoke_seconds": total_inference_seconds,
        "total_inference_seconds": total_inference_seconds,
        "per_sample_seconds": total_inference_seconds / float(batch_size),
        "batch_size": int(batch_size),
    }
    metrics.update(calc_metrics_nd(metric_clean, metric_noisy, metric_denoised))
    if sample_type is not None:
        metrics["sample_type"] = sample_type
        metrics["clean_detect_cc"] = float(clean_detect_cc)
    dataset_summary = summarize_precomputed_metrics(config)
    if dataset_summary is not None:
        metrics["dataset_summary"] = dataset_summary

    return {
        "config": config,
        "sample_index": sample_index,
        "model_path": str(model_path),
        "noisy": primary_noisy,
        "clean": primary_clean,
        "denoised": primary_denoised,
        "noisy_batch": noisy_batch,
        "clean_batch": clean_batch,
        "denoised_batch": denoised_batch,
        "metrics": metrics,
        "fs": config["default_fs"],
        "noisy_path": str(config["noisy_path"]),
        "clean_path": str(config["clean_path"]),
    }


def evaluate_autoencoder_full(config):
    model_path = resolve_model_path(config["model_path"])
    noisy_all = to_2d(np.load(config["noisy_path"]))
    clean_all = to_2d(np.load(config["clean_path"]))
    total_n = min(len(noisy_all), len(clean_all))

    interpreter = tflite.Interpreter(model_path=str(model_path), num_threads=NUM_THREADS)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    target_len = int(input_details["shape"][1])

    single_times = []
    decoded = []
    overall_t0 = time.perf_counter()
    for i in range(total_n):
        noisy = resample_1d(noisy_all[i], target_len).astype(np.float32)
        if len(input_details["shape"]) == 3:
            input_tensor = noisy.reshape(1, target_len, 1)
        else:
            input_tensor = noisy.reshape(1, target_len)
        interpreter.set_tensor(input_details["index"], input_tensor)
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        single_times.append(t1 - t0)
        decoded.append(to_2d(interpreter.get_tensor(output_details["index"]))[0])
    overall_t1 = time.perf_counter()

    decoded = np.asarray(decoded, dtype=np.float32)
    clean_rs = np.stack([resample_1d(x, target_len) for x in clean_all[:total_n]], axis=0)
    noisy_rs = np.stack([resample_1d(x, target_len) for x in noisy_all[:total_n]], axis=0)
    z_clean = np.stack([zero_center(x) for x in clean_rs], axis=0)
    z_decoded = np.stack([zero_center(x) for x in decoded], axis=0)

    groups = {
        "clean": {"rrmse_time": [], "rrmse_freq": [], "cc": []},
        "EOG": {"rrmse_time": [], "rrmse_freq": [], "cc": []},
        "Motion": {"rrmse_time": [], "rrmse_freq": [], "cc": []},
        "EMG": {"rrmse_time": [], "rrmse_freq": [], "cc": []},
    }
    for i in range(total_n):
        sample_type, _ = autoencoder_sample_type(clean_rs[i], noisy_rs[i], i)
        _, gt_psd = welch_psd(z_clean[i], fs=config["default_fs"])
        _, pred_psd = welch_psd(z_decoded[i], fs=config["default_fs"])
        groups[sample_type]["rrmse_time"].append(rrmse(z_clean[i], z_decoded[i]))
        groups[sample_type]["rrmse_freq"].append(rrmse(gt_psd, pred_psd))
        groups[sample_type]["cc"].append(pearson_corr(z_clean[i], z_decoded[i]))

    group_stats = {}
    for name, metrics in groups.items():
        group_stats[name] = {
            "rrmse_time": summarize_metric(metrics["rrmse_time"]),
            "rrmse_freq": summarize_metric(metrics["rrmse_freq"]),
            "cc": summarize_metric(metrics["cc"]),
        }

    return {
        "model_path": str(model_path),
        "noisy_path": str(config["noisy_path"]),
        "clean_path": str(config["clean_path"]),
        "total_samples": int(total_n),
        "target_len": int(target_len),
        "total_inference_seconds": float(overall_t1 - overall_t0),
        "per_sample_seconds": float(np.mean(single_times)) if single_times else 0.0,
        "group_stats": group_stats,
    }


def evaluate_precomputed_full(config):
    summary = summarize_precomputed_metrics(config)
    if summary is None:
        raise ValueError("This model does not provide precomputed full-dataset outputs.")

    return {
        "model_path": str(resolve_model_path(config["model_path"])),
        "noisy_path": str(config["noisy_path"]),
        "clean_path": str(config["clean_path"]),
        "denoised_path": str(config["precomputed_denoised_path"]),
        "summary": summary,
        "channels": len(config.get("channel_names", [])),
        "family": config["family"],
    }


def align_to_common_length(series_list):
    common_len = max(len(np.asarray(x).reshape(-1)) for x in series_list)
    return [resample_1d(x, common_len) for x in series_list], common_len


def plot_single_result(result, title_text):
    clean = result["clean"]
    noisy = result["noisy"]
    denoised = result["denoised"]
    fs = result["fs"]
    t = np.arange(clean.shape[0], dtype=np.float32) / float(fs)
    channel_names = result["config"].get("channel_names", [])

    if clean.ndim == 2 and clean.shape[1] > 1:
        n_channels = clean.shape[1]
        ncols = 2
        nrows = int(math.ceil(n_channels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=True)
        axes = np.atleast_1d(axes).reshape(-1)
        fig.suptitle(title_text, fontsize=13, fontweight="bold")
        for ch in range(n_channels):
            ax = axes[ch]
            label = channel_names[ch] if ch < len(channel_names) else f"Ch {ch + 1}"
            ax.plot(t, clean[:, ch], label="Clean", linewidth=1.0, alpha=0.45, color="black")
            ax.plot(t, noisy[:, ch], label="Noisy", linewidth=0.9, alpha=0.65, linestyle="--", color="tab:red")
            ax.plot(t, denoised[:, ch], label="Denoised", linewidth=1.0, color="tab:blue")
            ylim = auto_ylim([clean[:, ch], noisy[:, ch], denoised[:, ch]], family=result["config"]["family"])
            if ylim is not None:
                ax.set_ylim(*ylim)
            ch_metrics = calc_metrics_nd(clean[:, ch], noisy[:, ch], denoised[:, ch])
            ax.set_title(f"{label} | RRMSE {ch_metrics['rrmse']:.3f} | CC {ch_metrics['pearson_cc']:.3f}", fontsize=10)
            ax.grid(True, alpha=0.3)
        for ax in axes[n_channels:]:
            ax.axis("off")
        axes[0].legend(loc="upper right")
        fig.text(0.5, 0.04, "Time (s)", ha="center")
        fig.text(0.04, 0.5, "Amplitude", va="center", rotation="vertical")
        plt.tight_layout(rect=[0.04, 0.05, 1, 0.94])
        plt.show()
        return

    clean_1d = clean.reshape(-1)
    noisy_1d = noisy.reshape(-1)
    denoised_1d = denoised.reshape(-1)
    plt.figure(figsize=(11, 5))
    plt.plot(t, clean_1d, label="Clean", linewidth=1.2)
    plt.plot(t, noisy_1d, label="Noisy", linewidth=1.0, alpha=0.85)
    plt.plot(t, denoised_1d, label="Denoised", linewidth=1.1)
    ylim = auto_ylim([clean_1d, noisy_1d, denoised_1d], family=result["config"]["family"])
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title_text)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_batch4_grid(result, title_text):
    fs = result["fs"]
    clean_batch = result["clean_batch"]
    noisy_batch = result["noisy_batch"]
    denoised_batch = result["denoised_batch"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False)
    axes = axes.reshape(-1)
    for i, ax in enumerate(axes[:len(clean_batch)]):
        sample_clean = clean_batch[i]
        sample_noisy = noisy_batch[i]
        sample_denoised = denoised_batch[i]
        t = np.arange(sample_clean.shape[0], dtype=np.float32) / float(fs)
        if sample_clean.ndim == 2 and sample_clean.shape[1] > 1:
            sample_clean = sample_clean[:, 0]
            sample_noisy = sample_noisy[:, 0]
            sample_denoised = sample_denoised[:, 0]
            ax.set_title(f"Batch sample {i} | Ch 1")
        else:
            sample_clean = sample_clean.reshape(-1)
            sample_noisy = sample_noisy.reshape(-1)
            sample_denoised = sample_denoised.reshape(-1)
            ax.set_title(f"Batch sample {i}")
        ax.plot(t, sample_clean, label="Clean", linewidth=1.1)
        ax.plot(t, sample_noisy, label="Noisy", linewidth=0.95, alpha=0.85)
        ax.plot(t, sample_denoised, label="Denoised", linewidth=1.0)
        ylim = auto_ylim([sample_clean, sample_noisy, sample_denoised], family=result["config"]["family"])
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title_text)
    fig.tight_layout()
    plt.show()


def plot_multi_compare(results, title_text):
    first_clean = results[0]["clean"]
    if first_clean.ndim == 2 and first_clean.shape[1] > 1:
        channel_names = results[0]["config"].get("channel_names", [])
        common_len = max(r["clean"].shape[0] for r in results)
        n_channels = first_clean.shape[1]
        t = np.arange(common_len, dtype=np.float32) / float(results[0]["fs"])
        ncols = 2
        nrows = int(math.ceil(n_channels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), sharex=True)
        axes = np.atleast_1d(axes).reshape(-1)
        fig.suptitle(title_text, fontsize=13, fontweight="bold")

        for ch in range(n_channels):
            ax = axes[ch]
            clean_ch = resample_1d(results[0]["clean"][:, ch], common_len)
            noisy_ch = resample_1d(results[0]["noisy"][:, ch], common_len)
            ax.plot(t, clean_ch, label="Clean", linewidth=1.0, alpha=0.45, color="black")
            ax.plot(t, noisy_ch, label="Noisy", linewidth=0.9, alpha=0.60, linestyle="--", color="tab:red")
            den_series = []
            for result in results:
                label = result["config"]["display_zh"] if CURRENT_LANG == "zh" else result["config"]["display_en"]
                den = resample_1d(result["denoised"][:, ch], common_len)
                den_series.append(den)
                ax.plot(t, den, linewidth=1.0, label=label)
            ylim = auto_ylim([clean_ch, noisy_ch] + den_series)
            if ylim is not None:
                ax.set_ylim(*ylim)
            label = channel_names[ch] if ch < len(channel_names) else f"Ch {ch + 1}"
            ax.set_title(label, fontsize=10)
            ax.grid(True, alpha=0.3)

        for ax in axes[n_channels:]:
            ax.axis("off")

        axes[0].legend(loc="upper right", fontsize=8)
        fig.text(0.5, 0.04, "Time (s)", ha="center")
        fig.text(0.04, 0.5, "Amplitude", va="center", rotation="vertical")
        plt.tight_layout(rect=[0.04, 0.05, 1, 0.94])
        plt.show()
        return

    base_series = [results[0]["clean"], results[0]["noisy"]]
    model_series = [r["denoised"] for r in results]
    aligned, common_len = align_to_common_length(base_series + model_series)
    clean_aligned = aligned[0]
    noisy_aligned = aligned[1]
    denoised_aligned = aligned[2:]
    x = np.arange(common_len, dtype=np.float32)

    plt.figure(figsize=(12, 6))
    plt.plot(x, clean_aligned, label="Clean", linewidth=1.25)
    plt.plot(x, noisy_aligned, label="Noisy", linewidth=1.0, alpha=0.8)
    for result, den in zip(results, denoised_aligned):
        label = result["config"]["display_zh"] if CURRENT_LANG == 'zh' else result["config"]["display_en"]
        plt.plot(x, den, label=label, linewidth=1.05)
    families = {r["config"]["family"] for r in results}
    family_for_ylim = "autoencoder" if families == {"autoencoder"} else None
    ylim = auto_ylim([clean_aligned, noisy_aligned] + denoised_aligned, family=family_for_ylim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Aligned Sample Points")
    plt.ylabel("Amplitude")
    plt.title(title_text)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


class CompareApp:
    def __init__(self, root):
        self.root = root
        self.lang = tk.StringVar(value="zh")
        self.mode = tk.StringVar(value="single")
        self.sample_index = tk.StringVar(value="0")
        self.same_sample = tk.BooleanVar(value=True)
        self.enable_compare_plot = tk.BooleanVar(value=True)
        self.show_batch4 = tk.BooleanVar(value=False)
        self.single_model = tk.StringVar(value="fcnn_bnci8_large_ep100_200hz")
        self.last_result = None
        self._build_ui()
        self.refresh_ui_text()

    def tr(self, key):
        return I18N[self.lang.get()][key]

    def model_display(self, key):
        cfg = MODEL_CONFIGS[key]
        return cfg["display_zh"] if self.lang.get() == "zh" else cfg["display_en"]

    def _build_ui(self):
        self.root.geometry("1000x760")
        self.root.title("EEG Model Compare Tool")
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill="x")
        self.language_label = ttk.Label(top)
        self.language_label.grid(row=0, column=0, sticky="w")
        self.language_combo = ttk.Combobox(top, textvariable=self.lang, state="readonly", values=["zh", "en"], width=8)
        self.language_combo.grid(row=0, column=1, padx=(8, 16), sticky="w")
        self.language_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_ui_text())
        self.mode_label = ttk.Label(top)
        self.mode_label.grid(row=0, column=2, sticky="w")
        self.mode_combo = ttk.Combobox(top, textvariable=self.mode, state="readonly", values=["single", "multi"], width=12)
        self.mode_combo.grid(row=0, column=3, padx=(8, 16), sticky="w")
        self.mode_combo.bind("<<ComboboxSelected>>", lambda e: self.refresh_mode_visibility())
        self.sample_label = ttk.Label(top)
        self.sample_label.grid(row=0, column=4, sticky="w")
        self.sample_entry = ttk.Entry(top, textvariable=self.sample_index, width=10)
        self.sample_entry.grid(row=0, column=5, padx=(8, 0), sticky="w")

        body = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")

        self.single_frame = ttk.LabelFrame(left, padding=10)
        self.single_frame.pack(fill="x", pady=(0, 12))
        self.single_model_combo = ttk.Combobox(self.single_frame, textvariable=self.single_model, state="readonly", values=list(MODEL_CONFIGS.keys()), width=30)
        self.single_model_combo.pack(fill="x")

        self.multi_frame = ttk.LabelFrame(left, padding=10)
        self.multi_frame.pack(fill="x")
        self.models_label = ttk.Label(self.multi_frame)
        self.models_label.pack(anchor="w")
        self.model_listbox = tk.Listbox(self.multi_frame, selectmode=tk.MULTIPLE, height=8, exportselection=False)
        for key in MODEL_CONFIGS:
            self.model_listbox.insert(tk.END, key)
        self.model_listbox.pack(fill="x", pady=(6, 8))
        self.model_listbox.select_set(0, 1)
        self.same_sample_check = ttk.Checkbutton(self.multi_frame, variable=self.same_sample)
        self.same_sample_check.pack(anchor="w", pady=(4, 2))
        self.compare_plot_check = ttk.Checkbutton(self.multi_frame, variable=self.enable_compare_plot)
        self.compare_plot_check.pack(anchor="w", pady=(2, 2))
        self.show_batch4_check = ttk.Checkbutton(self.multi_frame, variable=self.show_batch4)
        self.show_batch4_check.pack(anchor="w", pady=(2, 6))

        button_row = ttk.Frame(left)
        button_row.pack(fill="x", pady=(12, 0))
        self.run_button = ttk.Button(button_row, command=self.run)
        self.run_button.pack(side="left")
        self.full_eval_button = ttk.Button(button_row, command=self.run_full_eval)
        self.full_eval_button.pack(side="left", padx=(8, 0))
        self.clear_button = ttk.Button(button_row, command=self.clear_output)
        self.clear_button.pack(side="left", padx=(8, 0))
        self.save_button = ttk.Button(button_row, command=self.save_last_result)
        self.save_button.pack(side="left", padx=(8, 0))

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))
        self.status_label = ttk.Label(right)
        self.status_label.pack(anchor="w", pady=(0, 6))
        self.note_label = ttk.Label(right, foreground="#666666")
        self.note_label.pack(anchor="w", pady=(0, 4))
        self.ae_note_label = ttk.Label(right, foreground="#666666")
        self.ae_note_label.pack(anchor="w", pady=(0, 6))
        self.output_text = tk.Text(right, wrap="word", font=("Consolas", 10))
        self.output_text.pack(fill="both", expand=True)
        self.refresh_mode_visibility()
        self.refresh_model_labels()

    def refresh_model_labels(self):
        self.single_model_combo["values"] = list(MODEL_CONFIGS.keys())
        existing_selection = self.model_listbox.curselection()
        self.model_listbox.delete(0, tk.END)
        for key in MODEL_CONFIGS:
            self.model_listbox.insert(tk.END, f"{key} | {self.model_display(key)}")
        if existing_selection:
            for idx in existing_selection:
                if idx < self.model_listbox.size():
                    self.model_listbox.select_set(idx)

    def refresh_ui_text(self):
        global CURRENT_LANG
        CURRENT_LANG = self.lang.get()
        self.root.title(self.tr("title"))
        self.language_label.config(text=self.tr("language"))
        self.mode_label.config(text=self.tr("mode"))
        self.sample_label.config(text=self.tr("sample_index"))
        self.single_frame.config(text=self.tr("single_mode"))
        self.multi_frame.config(text=self.tr("multi_mode"))
        self.models_label.config(text=self.tr("models"))
        self.same_sample_check.config(text=self.tr("same_sample"))
        self.compare_plot_check.config(text=self.tr("enable_compare_plot"))
        self.show_batch4_check.config(text=self.tr("show_batch4"))
        self.run_button.config(text=self.tr("run"))
        self.full_eval_button.config(text=self.tr("full_ae_eval"))
        self.clear_button.config(text=self.tr("clear"))
        self.save_button.config(text=self.tr("save_json"))
        self.status_label.config(text=self.tr("status_ready"))
        self.note_label.config(text=self.tr("comparison_note"))
        self.ae_note_label.config(text=self.tr("ae_note"))
        self.refresh_model_labels()

    def refresh_mode_visibility(self):
        if self.mode.get() == "single":
            self.single_frame.pack(fill="x", pady=(0, 12))
            self.multi_frame.pack_forget()
        else:
            self.single_frame.pack_forget()
            self.multi_frame.pack(fill="x")

    def append_text(self, text):
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)

    def clear_output(self):
        self.output_text.delete("1.0", tk.END)
        self.status_label.config(text=self.tr("status_ready"))

    def selected_multi_keys(self):
        indices = self.model_listbox.curselection()
        return [list(MODEL_CONFIGS.keys())[i] for i in indices]

    def validate_sample_index(self):
        try:
            value = int(self.sample_index.get().strip())
        except ValueError:
            raise ValueError(self.tr("error_sample"))
        if value < 0:
            raise ValueError(self.tr("error_sample"))
        return value

    def ensure_files_exist(self, keys):
        for key in keys:
            cfg = MODEL_CONFIGS[key]
            resolve_model_path(cfg["model_path"])
            if not cfg["noisy_path"].exists():
                raise FileNotFoundError(str(cfg["noisy_path"]))
            if not cfg["clean_path"].exists():
                raise FileNotFoundError(str(cfg["clean_path"]))

    def format_result_block(self, key, result):
        name = self.model_display(key)
        m = result["metrics"]
        lines = [
            f"[{name}]",
            f"model: {result['model_path']}",
            f"noisy_file: {result['noisy_path']}",
            f"clean_file: {result['clean_path']}",
            f"sample_index: {result['sample_index']}",
            f"batch_size_used: {m['batch_size']}",
            f"length: {m['length']}",
            f"channels: {m['channels']}",
            f"total_inference_seconds: {m['total_inference_seconds']:.6f}",
            f"per_sample_seconds: {m['per_sample_seconds']:.6f}",
            f"invoke_seconds: {m['invoke_seconds']:.6f}",
            f"rrmse: {m['rrmse']:.6f}",
            f"rmse: {m['rmse']:.6f}",
            f"pearson_cc_global: {m['pearson_cc']:.6f}",
            f"pearson_cc_channel_mean: {m['channel_mean_cc']:.6f}",
            f"pearson_cc_channel_min: {m['channel_min_cc']:.6f}",
            f"pearson_cc_channel_max: {m['channel_max_cc']:.6f}",
            f"baseline_rrmse: {m['baseline_rrmse']:.6f}",
            f"baseline_pearson_cc_global: {m['baseline_pearson_cc']:.6f}",
            f"baseline_pearson_cc_channel_mean: {m['baseline_channel_mean_cc']:.6f}",
            f"baseline_pearson_cc_channel_min: {m['baseline_channel_min_cc']:.6f}",
            f"baseline_pearson_cc_channel_max: {m['baseline_channel_max_cc']:.6f}",
        ]
        if "dataset_summary" in m:
            ds = m["dataset_summary"]
            lines.extend([
                "dataset_mean_metrics:",
                f"  count: {ds['count']}",
                f"  baseline_rrmse_mean: {ds['baseline_rrmse_mean']:.6f}",
                f"  rrmse_mean: {ds['rrmse_mean']:.6f}",
                f"  baseline_pearson_cc_global_mean: {ds['baseline_pearson_cc_mean']:.6f}",
                f"  pearson_cc_global_mean: {ds['pearson_cc_mean']:.6f}",
                f"  baseline_pearson_cc_channel_mean: {ds['baseline_channel_mean_cc_mean']:.6f}",
                f"  pearson_cc_channel_mean: {ds['channel_mean_cc_mean']:.6f}",
                f"  baseline_pearson_cc_channel_min_mean: {ds['baseline_channel_min_cc_mean']:.6f}",
                f"  pearson_cc_channel_min_mean: {ds['channel_min_cc_mean']:.6f}",
                f"  baseline_pearson_cc_channel_max_mean: {ds['baseline_channel_max_cc_mean']:.6f}",
                f"  pearson_cc_channel_max_mean: {ds['channel_max_cc_mean']:.6f}",
            ])
        if "sample_type" in m:
            lines.append(f"sample_type: {m['sample_type']}")
        if "clean_detect_cc" in m:
            lines.append(f"clean_detect_cc: {m['clean_detect_cc']:.6f}")
        return "\n".join(lines)

    def format_full_ae_eval_block(self, key, result):
        name = self.model_display(key)
        lines = [
            f"[{name} | full autoencoder evaluation]",
            f"model: {result['model_path']}",
            f"noisy_file: {result['noisy_path']}",
            f"clean_file: {result['clean_path']}",
            f"total_samples: {result['total_samples']}",
            f"length: {result['target_len']}",
            f"total_inference_seconds: {result['total_inference_seconds']:.6f}",
            f"per_sample_seconds: {result['per_sample_seconds']:.6f}",
        ]
        for group_name in ["clean", "EOG", "Motion", "EMG"]:
            g = result["group_stats"][group_name]
            lines.extend([
                "",
                f"{group_name}:",
                f"  RRMSE-Time mean={g['rrmse_time']['mean']:.4f}, std={g['rrmse_time']['std']:.4f}, n={g['rrmse_time']['count']}",
                f"  RRMSE-Freq mean={g['rrmse_freq']['mean']:.4f}, std={g['rrmse_freq']['std']:.4f}, n={g['rrmse_freq']['count']}",
                f"  CC mean={g['cc']['mean']:.4f}, std={g['cc']['std']:.4f}, n={g['cc']['count']}",
            ])
        return "\n".join(lines)

    def format_full_precomputed_eval_block(self, key, result):
        name = self.model_display(key)
        ds = result["summary"]
        lines = [
            f"[{name} | full dataset evaluation]",
            f"model: {result['model_path']}",
            f"noisy_file: {result['noisy_path']}",
            f"clean_file: {result['clean_path']}",
            f"denoised_file: {result['denoised_path']}",
            f"total_samples: {ds['count']}",
            f"baseline_rrmse_mean: {ds['baseline_rrmse_mean']:.6f}",
            f"rrmse_mean: {ds['rrmse_mean']:.6f}",
            f"baseline_pearson_cc_global_mean: {ds['baseline_pearson_cc_mean']:.6f}",
            f"pearson_cc_global_mean: {ds['pearson_cc_mean']:.6f}",
            f"baseline_pearson_cc_channel_mean: {ds['baseline_channel_mean_cc_mean']:.6f}",
            f"pearson_cc_channel_mean: {ds['channel_mean_cc_mean']:.6f}",
            f"baseline_pearson_cc_channel_min_mean: {ds['baseline_channel_min_cc_mean']:.6f}",
            f"pearson_cc_channel_min_mean: {ds['channel_min_cc_mean']:.6f}",
            f"baseline_pearson_cc_channel_max_mean: {ds['baseline_channel_max_cc_mean']:.6f}",
            f"pearson_cc_channel_max_mean: {ds['channel_max_cc_mean']:.6f}",
        ]
        return "\n".join(lines)


    def run(self):
        try:
            sample_index = self.validate_sample_index()
            keys = [self.single_model.get()] if self.mode.get() == "single" else self.selected_multi_keys()
            if not keys:
                raise ValueError(self.tr("error_select_model"))
            self.ensure_files_exist(keys)
        except Exception as exc:
            if isinstance(exc, FileNotFoundError):
                messagebox.showerror(self.tr("error_missing_file"), str(exc))
            else:
                messagebox.showerror("Error", str(exc))
            return

        self.status_label.config(text=self.tr("status_running"))
        self.root.update_idletasks()
        try:
            results = []
            for key in keys:
                result = run_inference_for_sample(MODEL_CONFIGS[key], sample_index)
                results.append((key, result))
                self.append_text(self.format_result_block(key, result))

            self.last_result = {
                "mode": self.mode.get(),
                "sample_index": sample_index,
                "results": [
                    {
                        "key": key,
                        "display_en": MODEL_CONFIGS[key]["display_en"],
                        "display_zh": MODEL_CONFIGS[key]["display_zh"],
                        "metrics": result["metrics"],
                        "model_path": result["model_path"],
                        "noisy_path": result["noisy_path"],
                        "clean_path": result["clean_path"],
                    }
                    for key, result in results
                ],
            }

            if self.mode.get() == "single":
                key, result = results[0]
                if result["metrics"]["batch_size"] == 4 and self.show_batch4.get():
                    plot_batch4_grid(result, f"{self.tr('single_plot_title')} | {self.model_display(key)} | k={sample_index}..{sample_index+3}")
                else:
                    plot_single_result(result, f"{self.tr('single_plot_title')} | {self.model_display(key)} | k={sample_index}")
            else:
                same_channel_count = len({r["metrics"]["channels"] for _, r in results}) == 1
                if self.enable_compare_plot.get() and len(results) >= 2 and same_channel_count:
                    plot_multi_compare([r for _, r in results], self.tr("compare_plot_title"))
                else:
                    if self.enable_compare_plot.get() and len(results) >= 2 and not same_channel_count:
                        self.append_text("compare_plot skipped: selected models do not share the same channel count.")
                    for key, result in results:
                        if result["metrics"]["batch_size"] == 4 and self.show_batch4.get():
                            plot_batch4_grid(result, f"{self.tr('single_plot_title')} | {self.model_display(key)} | k={sample_index}..{sample_index+3}")
                        else:
                            plot_single_result(result, f"{self.tr('single_plot_title')} | {self.model_display(key)} | k={sample_index}")

            self.status_label.config(text=self.tr("done"))
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.status_label.config(text=self.tr("status_ready"))

    def run_full_eval(self):
        key = self.single_model.get() if self.mode.get() == "single" else None
        if key is None:
            messagebox.showinfo("Info", "Please switch to single-model mode and choose one model.")
            return
        try:
            self.ensure_files_exist([key])
            self.status_label.config(text=self.tr("status_running"))
            self.root.update_idletasks()
            cfg = MODEL_CONFIGS[key]
            if cfg["family"] == "autoencoder":
                result = evaluate_autoencoder_full(cfg)
                self.append_text(self.format_full_ae_eval_block(key, result))
            elif cfg.get("precomputed_denoised_path") is not None:
                result = evaluate_precomputed_full(cfg)
                self.append_text(self.format_full_precomputed_eval_block(key, result))
            else:
                messagebox.showinfo("Info", "This model does not support full-dataset evaluation.")
                self.status_label.config(text=self.tr("status_ready"))
                return
            self.append_text("")
            self.status_label.config(text=self.tr("done"))
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.status_label.config(text=self.tr("status_ready"))

    def save_last_result(self):
        if not self.last_result:
            messagebox.showinfo("Info", "No result yet.")
            return
        out_path = OUTPUT_DIR / "metrics" / "last_result.json"
        out_path.write_text(json.dumps(self.last_result, indent=2, ensure_ascii=False), encoding="utf-8")
        messagebox.showinfo("Info", str(out_path))


def main():
    root = tk.Tk()
    app = CompareApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

