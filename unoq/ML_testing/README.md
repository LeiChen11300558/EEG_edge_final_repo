# ML_testing

This folder contains the Arduino UNO Q inference scripts for the autoencoder-style deployment path.

## Main files

- `run_tflite_float_unoq_v3_4core.py`: inference and evaluation script with timing output.
- `run_tflite_float_unoq_v3only_4core.py`: inference-only timing script.
- `run_tflite_float_unoq_v3self_4core.py`: transfer-learning model evaluation script.
- `run_tflite_float_unoq_v3selfonly_4core.py`: transfer-learning inference-only timing script.
- `bridge_pipeline.py`
- `bridge_pipeline_check.py`
- `bridge_pipeline_real.py`
  Linux-side scripts used together with `../testbridge/` to validate the bridge and combined pipeline.

## Runtime files

- `daefloat`
- `daefloatself`
- `x_test_noisy1.npy`
- `x_test_clean1.npy`

These files are required by the current scripts in this folder.

## Requirements

Install:

- `numpy`
- `tflite-runtime`

The local environment file is `requirements.txt`. The older file `requirements_mltest1.txt` is kept as part of the original project workflow.
