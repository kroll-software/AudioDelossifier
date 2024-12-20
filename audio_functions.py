import os
import numpy as np
from numpy.core.arrayprint import dtype_is_implied

import tensorflow as tf
# VS-Code Python debugger can't resolve this, but should still work
#from tensorflow.keras import layers
# instead you could 'pip install keras' and use
#from keras import layers

from tensorflow.python.platform.tf_logging import fatal
import tensorflow_io as tfio

import pydub
import scipy.io.wavfile
import soundfile as sf
#import librosa

import audio_config as cfg


def write_audiofile(data, filepath):
    int_data = data[0:]
    length = len(int_data)
    assert(length > 0)

    int_data = tf.scalar_mul(32767, int_data)
    int_data = tf.cast(int_data, tf.int16)

    encoded = tfio.audio.encode_wav(int_data, 44100)
    newFileByteArray = bytearray(encoded.numpy())
    
    print(f"Writing file '{filepath}'")
    with open(filepath, "wb") as newFile:
        newFile.write(newFileByteArray)

def write_audiofile_wav(data, filepath):
    int_data = data[0:]
    length = len(int_data)
    assert(length > 0)
    #int_data = int_data * 32767
    #int_data = int_data.astype(np.int16)
    print(f"Writing file '{filepath}'")
    scipy.io.wavfile.write(filepath, 44100, int_data)

def write_audiofile_flac(data, filepath):
    int_data = data[0:]
    length = len(int_data)
    assert(length > 0)
    #int_data = int_data * 32767
    #int_data = int_data.astype(np.int16)
    print(f"Writing file '{filepath}'")
    sf.write(filepath, int_data, samplerate=44100)
    
def read_audiofile(filepath):
    filename, file_extension = os.path.splitext(filepath)
    if file_extension == ".mp3":
        return read_audiofile_mp3(filepath)
    elif file_extension == ".flac":
        return read_audiofile_flac(filepath)
    else:
        return read_audiofile_wav(filepath)

def read_audiofile_wav(filepath):
    print(f"Reading audio file '{filepath}'")
    sanplerate, data = scipy.io.wavfile.read(filepath)
    assert sanplerate == 44100

    if data.dtype == np.int16:
        data = data.astype(np.float32)
        data = data * (1.0 / 32767.0)

    return data

def read_audiofile_flac(filepath):
    print(f"Reading audio file '{filepath}'")
    a, sanplerate = sf.read(filepath)
    assert sanplerate == 44100
    data = np.array(a, dtype=np.float32)
    return data

def read_audiofile_mp3(filepath):
    print(f"Reading audio file '{filepath}'")
    a = pydub.AudioSegment.from_mp3(filepath)
    data = np.array(a.get_array_of_samples())
    if a.channels == 2:
        data = data.reshape((-1, 2))
    if data.dtype == np.int16:
        data = data.astype(np.float32)
        data = data * (1.0 / 32767.0)     
    return data


def calc_mse(d1, d2):
    min_len = min(len(d1), len(d2))
    d1 = d1[:min_len]
    d2 = d2[:min_len]
    return np.mean((d1 - d2) ** 2)

def calc_mae(d1, d2):
    min_len = min(len(d1), len(d2))
    d1 = d1[:min_len]
    d2 = d2[:min_len]    
    return np.mean(np.abs((d1 - d2)))

def calc_accuracy(d1, d2):
    min_len = min(len(d1), len(d2))
    d1 = d1[:min_len]
    d2 = d2[:min_len]    
    correct = (d1 == d2)
    return correct.sum() / correct.size