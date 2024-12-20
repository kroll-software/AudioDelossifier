# Audio Delossifier

Delossify compressed audio (mp3 and others) with Python and Tensorflow.

This repository attempts to improve the audio quality of lossy compression such as MP3.

It is unclear whether this always succeeds. Perhaps you can use this project as a starting point for your own developments or improvements.


## Installation

This code was developed with Python 3.12.7.

Beside the python packages in requirements.txt you also need ffmpeg (on linux: 'apt-get install ffmpeg')

To install Tensorflow with GPU-support, please follow the instructions from the tensorflow website.

It is recommended to use a virtual environment such as venv or conda.


## Training

1. Copy some uncompressed audio files (*.wav or *.flac) to the /training-data/uncompressed folder
2. Set the Mp3BitRate and other configurations in audio_config.py
3. Delete the corresponding weights-file in /models when you want to train from scratch
4. Run audio-train.py

This repository comes with pre-trained models for some typical mp3-bitrates.


## Inference (delossify files)

1. Copy some compressed audio files (*.mp3 or *.wav) to the /audio folder
2. Set the OUTPUTFORMAT to 'FLAC' or 'WAV' in audio_config.py
3. Run audio_predict.py
4. The delossified files can be found in the /audio/out folder


### To-Do

Better analysis and benchmarks. We currently use MSE to compare the results, however for higher bitrates the MSE between compressed and uncompressed audio is already very low and might be below the accuracy of the ML computations.
