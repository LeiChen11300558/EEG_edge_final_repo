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
        "display_en": "Autoencoder After Transfer Learning",
        "display_zh": "Autoencoder After Transfer Learning",
        "model_path": MODEL_DIR / "autoencoder" / "daefloatself",
        "noisy_path": DATA_DIR / "autoencoder" / "x_test_noisy1.npy",
        "clean_path": DATA_DIR / "autoencoder" / "x_test_clean1.npy",
        "default_fs": 200.0,
        "fixed_batch": None,
    },
}


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
        "full_eval": "Run Full Dataset Eval",
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
        "full_eval": "??????",
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
    arr = np.asarray(arr, dtype=np.float32).ravel()
    if arr.size == 0:
        return 0.0
    return math.sqrt(float(np.sum(arr ** 2)) / float(arr.size))


def rmse(true, pred):
    true = np.asarray(true, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    return rms_value(true - pred)


def rrmse(true, pred):
    den = rms_value(true)
    if den == 0.0:
        return 0.0
    return rmse(true, pred) / den


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()
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
    arr = np.asarray(values, dtype=np.float32)
    return {"count": int(arr.size), "mean": float(np.mean(arr)), "std": float(np.std(arr))}


def run_inference_for_sample(config, sample_index):
    model_path = resolve_model_path(config["model_path"])
    noisy_all = to_2d(np.load(config["noisy_path"]))
    clean_all = to_2d(np.load(config["clean_path"]))
    total_n = min(len(noisy_all), len(clean_all))
    if sample_index < 0 or sample_index >= total_n:
        raise IndexError(f"sample_index={sample_index}, total_n={total_n}")

    interpreter = tflite.Interpreter(model_path=str(model_path), num_threads=NUM_THREADS)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_shape = input_details["shape"]
    target_len = int(input_shape[1])
    fixed_batch = config["fixed_batch"]
    batch_size = fixed_batch or 1
    if sample_index + batch_size > total_n:
        raise IndexError(f"Need {batch_size} consecutive samples starting at {sample_index}, but dataset has {total_n} samples")

    noisy_batch_raw = noisy_all[sample_index:sample_index + batch_size].astype(np.float32)
    clean_batch_raw = clean_all[sample_index:sample_index + batch_size].astype(np.float32)
    noisy_batch = np.stack([resample_1d(x, target_len) for x in noisy_batch_raw], axis=0)
    clean_batch = np.stack([resample_1d(x, target_len) for x in clean_batch_raw], axis=0)

    if fixed_batch is not None:
        input_tensor = noisy_batch.astype(np.float32)
    else:
        if len(input_shape) == 3:
            input_tensor = noisy_batch[:1].reshape(1, target_len, 1).astype(np.float32)
        else:
            input_tensor = noisy_batch[:1].reshape(1, target_len).astype(np.float32)
        clean_batch = clean_batch[:1]
        noisy_batch = noisy_batch[:1]
        batch_size = 1

    interpreter.set_tensor(input_details["index"], input_tensor)
    t0 = time.perf_counter()
    interpreter.invoke()
    t1 = time.perf_counter()
    output = interpreter.get_tensor(output_details["index"])
    denoised_batch = to_2d(output)[:batch_size].astype(np.float32)

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
        "invoke_seconds": total_inference_seconds,
        "total_inference_seconds": total_inference_seconds,
        "per_sample_seconds": total_inference_seconds / float(batch_size),
        "rmse": float(rmse(metric_clean, metric_denoised)),
        "rrmse": float(rrmse(metric_clean, metric_denoised)),
        "pearson_cc": float(pearson_corr(metric_clean, metric_denoised)),
        "baseline_rrmse": float(rrmse(metric_clean, metric_noisy)),
        "baseline_pearson_cc": float(pearson_corr(metric_clean, metric_noisy)),
        "batch_size": int(batch_size),
    }
    if sample_type is not None:
        metrics["sample_type"] = sample_type
        metrics["clean_detect_cc"] = float(clean_detect_cc)

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


def evaluate_model_full(config):
    model_path = resolve_model_path(config["model_path"])
    noisy_all = to_2d(np.load(config["noisy_path"]))
    clean_all = to_2d(np.load(config["clean_path"]))
    total_n = min(len(noisy_all), len(clean_all))
    fixed_batch = config["fixed_batch"]
    if fixed_batch not in (None, 1):
        raise ValueError("Full dataset evaluation currently supports single-sample models only.")

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
    metric_summary = {}

    if config["family"] == "autoencoder":
        z_test_noisy = np.zeros(noisy_rs.shape, dtype=np.float32)
        z_test_clean = np.zeros(clean_rs.shape, dtype=np.float32)
        z_decoded_layer = np.zeros(decoded.shape, dtype=np.float32)

        for i in range(total_n):
            z_test_noisy[i] = noisy_rs[i] - np.mean(noisy_rs[i])
            z_test_clean[i] = clean_rs[i] - np.mean(clean_rs[i])
            z_decoded_layer[i] = decoded[i].flatten() - np.mean(decoded[i].flatten())

        clean_detect = []
        noisy_detect = []
        cc_detect_clean = np.zeros(shape=(total_n, 1), dtype=np.float32)
        for i in range(total_n):
            cc_detect_clean[i] = np.corrcoef(z_test_clean[i], z_test_noisy[i])[0, 1]
            if cc_detect_clean[i] > 0.95:
                clean_detect.append(i)
            else:
                noisy_detect.append(i)

        clean_inputs = []
        clean_outputs = []
        noisy_inputs_eog = []
        noisy_inputs_motion = []
        noisy_inputs_emg = []
        noisy_outputs_eog = []
        noisy_outputs_motion = []
        noisy_outputs_emg = []
        ground_truth_eog = []
        ground_truth_motion = []
        ground_truth_emg = []

        for i in range(len(clean_detect)):
            clean_inputs.append(z_test_noisy[clean_detect[i]])
            clean_outputs.append(z_decoded_layer[clean_detect[i]])

        for i in range(len(noisy_detect)):
            sample_idx = noisy_detect[i]
            if sample_idx < 345:
                noisy_inputs_eog.append(z_test_noisy[sample_idx])
                noisy_outputs_eog.append(z_decoded_layer[sample_idx])
                ground_truth_eog.append(z_test_clean[sample_idx])
            elif sample_idx < 967:
                noisy_inputs_motion.append(z_test_noisy[sample_idx])
                noisy_outputs_motion.append(z_decoded_layer[sample_idx])
                ground_truth_motion.append(z_test_clean[sample_idx])
            else:
                noisy_inputs_emg.append(z_test_noisy[sample_idx])
                noisy_outputs_emg.append(z_decoded_layer[sample_idx])
                ground_truth_emg.append(z_test_clean[sample_idx])

        nperseg = 200
        nfft = 800
        psd_len = nfft // 2 + 1

        clean_inputs_rrmse = []
        clean_inputs_psd_rrmse = []
        clean_inputs_cc = []
        clean_inputs_psd = np.zeros(shape=(len(clean_inputs), psd_len), dtype=np.float32)
        clean_outputs_psd = np.zeros(shape=(len(clean_inputs), psd_len), dtype=np.float32)
        for i in range(len(clean_inputs)):
            clean_inputs_rrmse.append(rrmse(clean_inputs[i], clean_outputs[i]))
            _, pxx = welch_psd(clean_inputs[i], fs=config["default_fs"], nperseg=nperseg, nfft=nfft)
            clean_inputs_psd[i] = pxx
            _, pxx = welch_psd(clean_outputs[i], fs=config["default_fs"], nperseg=nperseg, nfft=nfft)
            clean_outputs_psd[i] = pxx
            clean_inputs_cc.append(pearson_corr(clean_inputs[i], clean_outputs[i]))
        for i in range(len(clean_inputs)):
            clean_inputs_psd_rrmse.append(rrmse(clean_inputs_psd[i], clean_outputs_psd[i]))

        eog_rrmse = []
        eog_psd_rrmse = []
        eog_cc = []
        ground_truth_eog_psd = np.zeros(shape=(len(noisy_inputs_eog), psd_len), dtype=np.float32)
        noisy_outputs_eog_psd = np.zeros(shape=(len(noisy_inputs_eog), psd_len), dtype=np.float32)
        for i in range(len(noisy_inputs_eog)):
            eog_rrmse.append(rrmse(ground_truth_eog[i], noisy_outputs_eog[i]))
            _, pxx = welch_psd(ground_truth_eog[i], fs=config["default_fs"], nperseg=nperseg, nfft=nfft)
            ground_truth_eog_psd[i] = pxx
            _, pxx = welch_psd(noisy_outputs_eog[i], fs=config["default_fs"], nperseg=nperseg, nfft=nfft)
            noisy_outputs_eog_psd[i] = pxx
            eog_cc.append(pearson_corr(ground_truth_eog[i], noisy_outputs_eog[i]))
        for i in range(len(noisy_inputs_eog)):
            eog_psd_rrmse.append(rrmse(ground_truth_eog_psd[i], noisy_outputs_eog_psd[i]))

        motion_rrmse = []
        motion_psd_rrmse = []
        motion_cc = []
        ground_truth_motion_psd = np.zeros(shape=(len(noisy_inputs_motion), psd_len), dtype=np.float32)
        noisy_outputs_motion_psd = np.zeros(shape=(len(noisy_inputs_motion), psd_len), dtype=np.float32)
        for i in range(len(noisy_inputs_motion)):
            motion_rrmse.append(rrmse(ground_truth_motion[i], noisy_outputs_motion[i]))
            _, pxx = welch_psd(ground_truth_motion[i], fs=config["default_fs"], nperseg=nperseg, nfft=nfft)
            ground_truth_motion_psd[i] = pxx
            _, pxx = welch_psd(noisy_outputs_motion[i], fs=config["default_fs"], nperseg=nperseg, nfft=nfft)
            noisy_outputs_motion_psd[i] = pxx
            motion_cc.append(pearson_corr(ground_truth_motion[i], noisy_outputs_motion[i]))
        for i in range(len(noisy_inputs_motion)):
            motion_psd_rrmse.append(rrmse(ground_truth_motion_psd[i], noisy_outputs_motion_psd[i]))

        emg_rrmse = []
        emg_psd_rrmse = []
        emg_cc = []
        ground_truth_emg_psd = np.zeros(shape=(len(noisy_inputs_emg), psd_len), dtype=np.float32)
        noisy_outputs_emg_psd = np.zeros(shape=(len(noisy_inputs_emg), psd_len), dtype=np.float32)
        for i in range(len(noisy_inputs_emg)):
            emg_rrmse.append(rrmse(ground_truth_emg[i], noisy_outputs_emg[i]))
            _, pxx = welch_psd(ground_truth_emg[i], fs=config["default_fs"], nperseg=nperseg, nfft=nfft)
            ground_truth_emg_psd[i] = pxx
            _, pxx = welch_psd(noisy_outputs_emg[i], fs=config["default_fs"], nperseg=nperseg, nfft=nfft)
            noisy_outputs_emg_psd[i] = pxx
            emg_cc.append(pearson_corr(ground_truth_emg[i], noisy_outputs_emg[i]))
        for i in range(len(noisy_inputs_emg)):
            emg_psd_rrmse.append(rrmse(ground_truth_emg_psd[i], noisy_outputs_emg_psd[i]))

        metric_summary = {
            "mode": "grouped_autoencoder",
            "group_stats": {
                "clean": {
                    "rrmse_time": summarize_metric(clean_inputs_rrmse),
                    "rrmse_freq": summarize_metric(clean_inputs_psd_rrmse),
                    "cc": summarize_metric(clean_inputs_cc),
                },
                "EOG": {
                    "rrmse_time": summarize_metric(eog_rrmse),
                    "rrmse_freq": summarize_metric(eog_psd_rrmse),
                    "cc": summarize_metric(eog_cc),
                },
                "Motion": {
                    "rrmse_time": summarize_metric(motion_rrmse),
                    "rrmse_freq": summarize_metric(motion_psd_rrmse),
                    "cc": summarize_metric(motion_cc),
                },
                "EMG": {
                    "rrmse_time": summarize_metric(emg_rrmse),
                    "rrmse_freq": summarize_metric(emg_psd_rrmse),
                    "cc": summarize_metric(emg_cc),
                },
            },
        }
    else:
        overall = {
            "rmse": [],
            "rrmse": [],
            "cc": [],
            "baseline_rrmse": [],
            "baseline_cc": [],
        }
        for i in range(total_n):
            overall["rmse"].append(rmse(clean_rs[i], decoded[i]))
            overall["rrmse"].append(rrmse(clean_rs[i], decoded[i]))
            overall["cc"].append(pearson_corr(clean_rs[i], decoded[i]))
            overall["baseline_rrmse"].append(rrmse(clean_rs[i], noisy_rs[i]))
            overall["baseline_cc"].append(pearson_corr(clean_rs[i], noisy_rs[i]))
        metric_summary = {
            "mode": "overall",
            "overall": {
                "rmse": summarize_metric(overall["rmse"]),
                "rrmse": summarize_metric(overall["rrmse"]),
                "cc": summarize_metric(overall["cc"]),
                "baseline_rrmse": summarize_metric(overall["baseline_rrmse"]),
                "baseline_cc": summarize_metric(overall["baseline_cc"]),
            },
        }

    return {
        "model_path": str(model_path),
        "noisy_path": str(config["noisy_path"]),
        "clean_path": str(config["clean_path"]),
        "family": config["family"],
        "total_samples": int(total_n),
        "target_len": int(target_len),
        "total_inference_seconds": float(overall_t1 - overall_t0),
        "per_sample_seconds": float(np.mean(single_times)) if single_times else 0.0,
        "min_per_sample_seconds": float(np.min(single_times)) if single_times else 0.0,
        "max_per_sample_seconds": float(np.max(single_times)) if single_times else 0.0,
        "first_per_sample_seconds": float(single_times[0]) if single_times else 0.0,
        "metric_summary": metric_summary,
    }


def align_to_common_length(series_list):
    common_len = max(len(np.asarray(x).reshape(-1)) for x in series_list)
    return [resample_1d(x, common_len) for x in series_list], common_len


def plot_single_result(result, title_text):
    clean = result["clean"]
    noisy = result["noisy"]
    denoised = result["denoised"]
    fs = result["fs"]
    t = np.arange(len(clean), dtype=np.float32) / float(fs)

    plt.figure(figsize=(11, 5))
    plt.plot(t, clean, label="Clean", linewidth=1.2)
    plt.plot(t, noisy, label="Noisy", linewidth=1.0, alpha=0.85)
    plt.plot(t, denoised, label="Denoised", linewidth=1.1)
    ylim = auto_ylim([clean, noisy, denoised], family=result["config"]["family"])
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
        t = np.arange(len(clean_batch[i]), dtype=np.float32) / float(fs)
        ax.plot(t, clean_batch[i], label="Clean", linewidth=1.1)
        ax.plot(t, noisy_batch[i], label="Noisy", linewidth=0.95, alpha=0.85)
        ax.plot(t, denoised_batch[i], label="Denoised", linewidth=1.0)
        ylim = auto_ylim([clean_batch[i], noisy_batch[i], denoised_batch[i]], family=result["config"]["family"])
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(f"Batch sample {i}")
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title_text)
    fig.tight_layout()
    plt.show()


def plot_multi_compare(results, title_text):
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
        self.single_model = tk.StringVar(value="fcnn_200hz")
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
        self.full_eval_button = ttk.Button(button_row, command=self.run_full_autoencoder_eval)
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
        self.full_eval_button.config(text=self.tr("full_eval"))
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
            f"total_inference_seconds: {m['total_inference_seconds']:.6f}",
            f"per_sample_seconds: {m['per_sample_seconds']:.6f}",
            f"invoke_seconds: {m['invoke_seconds']:.6f}",
            f"rrmse: {m['rrmse']:.6f}",
            f"rmse: {m['rmse']:.6f}",
            f"pearson_cc: {m['pearson_cc']:.6f}",
            f"baseline_rrmse: {m['baseline_rrmse']:.6f}",
            f"baseline_pearson_cc: {m['baseline_pearson_cc']:.6f}",
        ]
        if "sample_type" in m:
            lines.append(f"sample_type: {m['sample_type']}")
        if "clean_detect_cc" in m:
            lines.append(f"clean_detect_cc: {m['clean_detect_cc']:.6f}")
        return "\n".join(lines)

    def format_full_eval_block(self, key, result):
        summary = result["metric_summary"]
        if summary["mode"] == "grouped_autoencoder":
            clean_stats = summary["group_stats"]["clean"]
            eog_stats = summary["group_stats"]["EOG"]
            motion_stats = summary["group_stats"]["Motion"]
            emg_stats = summary["group_stats"]["EMG"]
            lines = [
                "Total inference time: {:.2f} seconds".format(result["total_inference_seconds"]),
                "Average inference time per segment: {:.4f} s ({:.2f} ms)".format(result["per_sample_seconds"], result["per_sample_seconds"] * 1000.0),
                "Min / Max per-segment time: {:.4f} s / {:.4f} s".format(result["min_per_sample_seconds"], result["max_per_sample_seconds"]),
                "First segment inference time: {:.4f} s ({:.2f} ms)".format(result["first_per_sample_seconds"], result["first_per_sample_seconds"] * 1000.0),
                "",
                " EEG clean input results: ",
                "RRMSE-Time: mean=  {:.4f}  ,std=  {:.4f}".format(clean_stats["rrmse_time"]["mean"], clean_stats["rrmse_time"]["std"]),
                "RRMSE-Freq: mean=  {:.4f}  ,std=  {:.4f}".format(clean_stats["rrmse_freq"]["mean"], clean_stats["rrmse_freq"]["std"]),
                "CC: mean=  {:.4f}  ,std=  {:.4f}".format(clean_stats["cc"]["mean"], clean_stats["cc"]["std"]),
                "",
                " EEG EOG artifacts results:",
                "RRMSE-Time: mean=  {:.4f}  ,std=  {:.4f}".format(eog_stats["rrmse_time"]["mean"], eog_stats["rrmse_time"]["std"]),
                "RRMSE-Freq: mean=  {:.4f}  ,std=  {:.4f}".format(eog_stats["rrmse_freq"]["mean"], eog_stats["rrmse_freq"]["std"]),
                "CC: mean=  {:.4f}  ,std=  {:.4f}".format(eog_stats["cc"]["mean"], eog_stats["cc"]["std"]),
                " ",
                " EEG motion artifacts results:",
                "RRMSE-Time:  mean=  {:.4f}  ,std=  {:.4f}".format(motion_stats["rrmse_time"]["mean"], motion_stats["rrmse_time"]["std"]),
                "RRMSE-Freq:  mean=  {:.4f}  ,std=  {:.4f}".format(motion_stats["rrmse_freq"]["mean"], motion_stats["rrmse_freq"]["std"]),
                "CC:  mean=  {:.4f}  ,std=  {:.4f}".format(motion_stats["cc"]["mean"], motion_stats["cc"]["std"]),
                " ",
                " EEG EMG artifacts results:",
                "RRMSE-Time:  mean=  {:.4f}  ,std=  {:.4f}".format(emg_stats["rrmse_time"]["mean"], emg_stats["rrmse_time"]["std"]),
                "RRMSE-Freq:  mean=  {:.4f}  ,std=  {:.4f}".format(emg_stats["rrmse_freq"]["mean"], emg_stats["rrmse_freq"]["std"]),
                "CC:  mean=  {:.4f}  ,std=  {:.4f}".format(emg_stats["cc"]["mean"], emg_stats["cc"]["std"]),
            ]
            return "\n".join(lines)

        overall = summary["overall"]
        lines = [
            f"[{self.model_display(key)} | full dataset evaluation]",
            f"model: {result['model_path']}",
            f"noisy_file: {result['noisy_path']}",
            f"clean_file: {result['clean_path']}",
            f"family: {result['family']}",
            f"total_samples: {result['total_samples']}",
            f"length: {result['target_len']}",
            f"total_inference_seconds: {result['total_inference_seconds']:.6f}",
            f"per_sample_seconds: {result['per_sample_seconds']:.6f}",
            "",
            "overall:",
            f"  RMSE mean={overall['rmse']['mean']:.4f}, std={overall['rmse']['std']:.4f}, n={overall['rmse']['count']}",
            f"  RRMSE mean={overall['rrmse']['mean']:.4f}, std={overall['rrmse']['std']:.4f}, n={overall['rrmse']['count']}",
            f"  CC mean={overall['cc']['mean']:.4f}, std={overall['cc']['std']:.4f}, n={overall['cc']['count']}",
            f"  Baseline RRMSE mean={overall['baseline_rrmse']['mean']:.4f}, std={overall['baseline_rrmse']['std']:.4f}, n={overall['baseline_rrmse']['count']}",
            f"  Baseline CC mean={overall['baseline_cc']['mean']:.4f}, std={overall['baseline_cc']['std']:.4f}, n={overall['baseline_cc']['count']}",
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
                if self.enable_compare_plot.get() and len(results) >= 2:
                    plot_multi_compare([r for _, r in results], self.tr("compare_plot_title"))
                else:
                    for key, result in results:
                        if result["metrics"]["batch_size"] == 4 and self.show_batch4.get():
                            plot_batch4_grid(result, f"{self.tr('single_plot_title')} | {self.model_display(key)} | k={sample_index}..{sample_index+3}")
                        else:
                            plot_single_result(result, f"{self.tr('single_plot_title')} | {self.model_display(key)} | k={sample_index}")

            self.status_label.config(text=self.tr("done"))
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.status_label.config(text=self.tr("status_ready"))

    def run_full_autoencoder_eval(self):
        key = self.single_model.get() if self.mode.get() == "single" else None
        supported_keys = {"autoencoder", "autoencoder_self", "fcnn_std", "fcnn_200hz"}
        if key is None or key not in supported_keys:
            messagebox.showinfo("Info", "Please switch to single-model mode and choose autoencoder, autoencoder_after_transfer_learning, fcnn_std, or fcnn_200hz.")
            return
        try:
            self.ensure_files_exist([key])
            self.status_label.config(text=self.tr("status_running"))
            self.root.update_idletasks()
            result = evaluate_model_full(MODEL_CONFIGS[key])
            self.append_text(self.format_full_eval_block(key, result))
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
