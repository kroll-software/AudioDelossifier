# Audio Delossifier
Delossify compressed audio (mp3 and others) with Python and Tensorflow.

Copy your 32-bit floating point *.wav files to the /audio folder.
Run audio_predict.py and find your delossified files in /audio/out

I use this together with my digital audio workstation (DAW), so all wav-files are supposed to be 32-bit floating point (for the input, it was converted from the compressed audio).

## Training

This repository comes with pre-trained models and can convert files without training.

To train a model, copy compressed audio as 32-bit floating point *.wav files to the /training-data/compressed folder.
Also copy the uncompressed versions with the same file names to the /training-data/uncompressed folder.

Then run audio_train.py. The script first tries to perfectly align the samples and caches the resultig files in /aligned-data. This might fail, when the files are stretched for some reason and can't be aligned.

Some settings can be configured in audio_config.py.

The LSTM runs in stateful-mode, so it should work even well with small chunks of data.