# fu8channel

This folder contains the 8-channel deployment scripts used for the later multi-channel extension on Arduino UNO Q.

## Main files

- `code/mainfu8_200hz.py`: evaluation script with metrics.
- `code/mainfu8_200hzonly.py`: inference-only timing script.
- `code/mainfu8_200hzone.py`: alternative evaluation script variant.
- `code/mainfu8_200hzoneonly.py`: alternative inference-only timing script variant.

## Runtime files

The scripts in this folder use:

- `code/fu8_large_200hz`
- `data/noiseinput_test.npy`
- `data/EEG_test.npy`

## Requirements

Install:

- `numpy`
- `tflite-runtime`

The local environment file is `requirements.txt`.
