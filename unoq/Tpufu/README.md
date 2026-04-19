# Tpufu

This folder contains the single-channel FCNN deployment scripts used on Arduino UNO Q.

## Main files

- `code/mainfu.py`: evaluation script with metrics.
- `code/mainfuonly.py`: inference-only timing script.
- `code/mainfu200hz.py`: 200 Hz evaluation script.
- `code/mainfu200hzonly.py`: 200 Hz inference-only timing script.
- `code/mainfu200hzbatch4.py`: batch-4 200 Hz evaluation script.
- `code/mainfu200hzbatch4only.py`: batch-4 200 Hz inference-only timing script.

## Runtime files

The scripts in `code/` currently use:

- model files in the same `code/` folder, such as `fu`, `fu200hz`, and `fu200hzbatch4`
- input arrays in `data/`, including `noiseinput_test.npy` and `EEG_test.npy`

## Requirements

Install:

- `numpy`
- `tflite-runtime`

The local environment file is `requirements.txt`.
