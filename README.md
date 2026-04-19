# Real-time EEG Artifact Removal and Edge Deployment Using Deep Learning

This repository contains the code used in the final year individual project on EEG denoising and Arduino UNO Q deployment. The project studies how deep learning models for EEG artifact removal can be reproduced, modified, and deployed on a low-power edge platform, with additional work on Linux--STM32 bridge communication and an 8-channel extension.

## Repository structure

- `pc/`: PC-side model development, training, offline evaluation, and supporting analysis.
- `unoq/`: Arduino UNO Q deployment code, board-side inference scripts, and Linux--STM32 bridge communication tests.

### PC-side folders

- `pc/Autoencoder_transferlearning/`: autoencoder reproduction, transfer learning, quantization, and TensorFlow Lite export.
- `pc/EEGdenoiseNetfuplot/`: single-channel FCNN reproduction and evaluation.
- `pc/EEGdenoiseNetfuplot200hz/`: 200 Hz FCNN adaptation and comparison.
- `pc/EEGdenoiseNetfuplot200hzmutiple_channel/`: multi-channel extension, including the later 8-channel study.

### UNO Q-side folders

- `unoq/ML_testing/`: autoencoder-related TensorFlow Lite inference scripts for Arduino UNO Q.
- `unoq/Tpufu/`: single-channel FCNN deployment and timing scripts on Arduino UNO Q.
- `unoq/fu8channel/`: 8-channel deployment scripts and related runtime files.
- `unoq/testbridge/`: Linux--STM32 bridge communication test project.

## Main code entry points

The most relevant scripts for the project are:

- `pc/Autoencoder_transferlearning/code_data/BigModel_v4_1.py`
  Autoencoder-side training and evaluation workflow used for reproduction and later transfer learning work.
- `pc/EEGdenoiseNetfuplot/code/benchmark_networks/main.py`
  Single-channel FCNN baseline workflow.
- `pc/EEGdenoiseNetfuplot200hz/code/benchmark_networks/main200hz.py`
  200 Hz FCNN workflow.
- `pc/EEGdenoiseNetfuplot200hzmutiple_channel/code/benchmark_networks/main.py`
  Multi-channel and 8-channel FCNN workflow.
- `unoq/ML_testing/run_tflite_float_unoq_v3_4core.py`
  Autoencoder TFLite inference on Arduino UNO Q.
- `unoq/Tpufu/code/mainfu.py`
  Single-channel FCNN deployment and evaluation on Arduino UNO Q.
- `unoq/Tpufu/code/mainfu200hz.py`
  200 Hz FCNN deployment on Arduino UNO Q.
- `unoq/fu8channel/code/mainfu8_200hz.py`
  8-channel deployment and evaluation script.
- `unoq/ML_testing/bridge_pipeline.py`
- `unoq/ML_testing/bridge_pipeline_check.py`
- `unoq/ML_testing/bridge_pipeline_real.py`
  Linux-side scripts used together with `unoq/testbridge/` to test and validate the bridge communication path.

## Environments and local setup files

The main environment files are:

- `pc/Autoencoder_transferlearning/requirements.txt`
- `pc/EEGdenoiseNetfuplot/requirements.txt`
- `pc/EEGdenoiseNetfuplot200hz/requirements.txt`
- `pc/EEGdenoiseNetfuplot200hzmutiple_channel/requirements.txt`
- `unoq/ML_testing/requirements.txt`
- `unoq/Tpufu/requirements.txt`
- `unoq/fu8channel/requirements.txt`
- `unoq/testbridge/python/requirements.txt`

The main local README files are:

- `pc/Autoencoder_transferlearning/README.md`
- `unoq/ML_testing/README.md`
- `unoq/Tpufu/README.md`
- `unoq/fu8channel/README.md`
- `unoq/testbridge/README.md`

For the FCNN PC-side folders, the existing folder-level README and requirements files should be used directly. For the UNO Q deployment folders, the local README files explain which scripts, models, and test arrays are used together.

## Upstream basis and project contribution

This repository contains both adapted open-source code and project-specific code written during the individual project.

### Code adapted from existing open-source repositories

- `pc/EEGdenoiseNetfuplot/`, `pc/EEGdenoiseNetfuplot200hz/`, and `pc/EEGdenoiseNetfuplot200hzmutiple_channel/`
  These folders are based on the EEGdenoiseNet repository and were modified in this project for FCNN experiments, plotting, 200 Hz adaptation, and the multi-channel extension.
- `pc/Autoencoder_transferlearning/`
  This folder is based on the Autoencoder repository and was modified in this project for reproduction, transfer learning, quantization, and TensorFlow Lite export.
- Parts of `unoq/ML_testing/`
  The inference-side implementation was adapted from the `autoencoder_coral_mini` example in the `tensorflow-lite-microcontroller-autoencoder` repository and then modified for Arduino UNO Q execution, timing measurement, and denoising evaluation.

### Code developed mainly in this project

- `unoq/Tpufu/`
  Arduino UNO Q deployment scripts for FCNN-based inference and timing evaluation.
- `unoq/fu8channel/`
  8-channel deployment-side scripts and tests.
- `unoq/testbridge/`
  Linux--STM32 bridge communication test project.
- `unoq/ML_testing/bridge_pipeline.py`
- `unoq/ML_testing/bridge_pipeline_check.py`
- `unoq/ML_testing/bridge_pipeline_real.py`
  These bridge pipeline scripts were written for this project and are used with `testbridge` to test the combined bridge and inference workflow.

The main project contribution lies in the modified model workflows, the Arduino UNO Q deployment scripts, the bridge communication implementation, and the later 8-channel extension.

## Running the code

The repository contains code from several stages of the project, so there is no single command that runs the whole project end to end. Instead, the folders are used for different parts of the workflow:

- Use `pc/` for training, offline evaluation, comparison plots, and preparation work.
- Use `unoq/ML_testing/` for autoencoder-style TFLite inference on Arduino UNO Q.
- Use `unoq/Tpufu/` for FCNN deployment on Arduino UNO Q.
- Use `unoq/testbridge/` together with the `bridge_pipeline` scripts in `unoq/ML_testing/` for Linux--STM32 bridge tests.
- Use `unoq/fu8channel/` for the 8-channel deployment experiments.

Please install the dependencies from the relevant local `requirements.txt` file before running each part of the project.

## Notes

- This repository includes code, runtime artefacts, and experiment files that were kept to match the practical workflow used during the project.
- Some folders still reflect the structure of the original upstream repositories, while others were added specifically for deployment and system integration.
- If this repository is used for public release, the upstream repositories and dataset sources should also be acknowledged clearly in the final published README.
