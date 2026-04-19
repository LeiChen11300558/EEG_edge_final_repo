# Model Compare Plot

This project is for local comparison and plotting across multiple EEG denoising models.

Planned model groups:
- Standard FCNN
- FCNN 200Hz
- FCNN 200Hz batch4
- Autoencoder

Suggested layout:
- `data/`: input EEG, noisy EEG, clean EEG, and any shared evaluation data
- `models/`: exported TFLite or Keras model files
- `scripts/`: loading, inference, metrics, and plotting scripts
- `outputs/figures/`: saved plots
- `outputs/metrics/`: saved CSV or JSON metric summaries

Next step:
- place model files and reference data into this folder structure
- then implement a unified local plotting script inspired by `code/benchmark_networks/plot_k.py`
