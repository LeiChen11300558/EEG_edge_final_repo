# Auto-generated helper for relative paths (POSIX-safe)
from pathlib import Path

# Project root: Autoencoder/
PROJECT = Path(__file__).resolve().parents[1]

# Folder containing this file (run_model/)
RUN_MODEL_DIR = Path(__file__).resolve().parent

# Common paths (Path objects)
MODEL_DIR = RUN_MODEL_DIR / "Autoencoder_revision"
TFLITE_REVISION = RUN_MODEL_DIR / "autoencoder_revision.tflite"
TFLITE_FLOAT16 = RUN_MODEL_DIR / "autoencoder_float16.tflite"
TFLITE_INT8    = RUN_MODEL_DIR / "autoencoder_int8.tflite"

X_TEST_NOISY = RUN_MODEL_DIR / "x_test_noisy1.npy"
X_TEST_CLEAN = RUN_MODEL_DIR / "x_test_clean1.npy"

# Convert Path to POSIX (forward-slash) string to avoid backslash escapes on Windows
def S(p: Path) -> str:
    return p.as_posix()

# Precomputed POSIX strings (optional)
MODEL_DIR_S       = S(MODEL_DIR)
TFLITE_REVISION_S = S(TFLITE_REVISION)
TFLITE_FLOAT16_S  = S(TFLITE_FLOAT16)
TFLITE_INT8_S     = S(TFLITE_INT8)
X_TEST_NOISY_S    = S(X_TEST_NOISY)
X_TEST_CLEAN_S    = S(X_TEST_CLEAN)
