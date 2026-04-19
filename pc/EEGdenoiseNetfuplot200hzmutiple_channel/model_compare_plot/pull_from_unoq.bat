@echo off
setlocal

set TARGET_ROOT=D:/pycharm file/EEGdenoiseNetfuplot200hz/model_compare_plot
set BOARD=arduino@192.168.1.243

echo Pulling FCNN models...
scp %BOARD%:/home/arduino/ArduinoApps/Tpufu/code/fu "%TARGET_ROOT%/models/fcnn"
scp %BOARD%:/home/arduino/ArduinoApps/Tpufu/code/fu200hz "%TARGET_ROOT%/models/fcnn"
scp %BOARD%:/home/arduino/ArduinoApps/Tpufu/code/fu200hzbatch4 "%TARGET_ROOT%/models/fcnn"

echo Pulling FCNN data...
scp %BOARD%:/home/arduino/ArduinoApps/Tpufu/data/noiseinput_test.npy "%TARGET_ROOT%/data/fcnn"
scp %BOARD%:/home/arduino/ArduinoApps/Tpufu/data/EEG_test.npy "%TARGET_ROOT%/data/fcnn"
scp %BOARD%:/home/arduino/ArduinoApps/Tpufu/data/EEG_all_epochs.npy "%TARGET_ROOT%/data/fcnn"
scp %BOARD%:/home/arduino/ArduinoApps/Tpufu/data/EOG_all_epochs.npy "%TARGET_ROOT%/data/fcnn"

echo Pulling autoencoder models...
scp %BOARD%:/home/arduino/ArduinoApps/autoencoderpy/ML_testing/daefloat "%TARGET_ROOT%/models/autoencoder"
scp %BOARD%:/home/arduino/ArduinoApps/autoencoderpy/ML_testing/daefloatself "%TARGET_ROOT%/models/autoencoder"

echo Pulling autoencoder data...
scp %BOARD%:/home/arduino/ArduinoApps/autoencoderpy/ML_testing/x_test_noisy1.npy "%TARGET_ROOT%/data/autoencoder"
scp %BOARD%:/home/arduino/ArduinoApps/autoencoderpy/ML_testing/x_test_clean1.npy "%TARGET_ROOT%/data/autoencoder"

echo Done.
pause
