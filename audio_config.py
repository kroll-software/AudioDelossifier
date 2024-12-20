import os

## inference
# copy some files to /audio and run audio_predict.py
OUTPUTFORMAT = "WAV"    # FLAC | WAV

# bitrate for training and inference
# will be used for inference on *.wav files
# will be automatically adjusted for inference on *.mp3 files
#Mp3BitRate = '64k'
#Mp3BitRate = '96k'
#Mp3BitRate = '128k'
Mp3BitRate = '192k'
#Mp3BitRate = '256k'
#Mp3BitRate = '320k'

## training
# copy some files to /uncompressed and run audio_train.py
DATAPATH = './training-data/'

BatchSize = 32  #decrease when you run out-of-memory (OOM)
LearningRate = 1e-5 #default is 1e-3, we have a scheduler
Epochs = 50

# model parameters
NumChannels = 2
InputSize = 2048

def nearest_bitrate(bitrate: str) -> str:
    b = int(bitrate[0: -1])
    if b > 288:
        return "320k"
    elif b > 224:
        return "256k"
    elif b > 160:
        return "192k"
    elif b > 112:
        return "128k"
    elif b > 80:
        return "96k"
    else:
        return "64k"


def ModelFileName(bitrate: str = None) -> str:
    if not bitrate: 
        bitrate = Mp3BitRate
    else:
        bitrate = nearest_bitrate(bitrate)
    return os.path.join("./models/", f'WaveletModel-{bitrate}-{InputSize}.weights.h5')