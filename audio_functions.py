import os
import pathlib
import numpy as np
import glob
from numpy.core.arrayprint import dtype_is_implied

import tensorflow as tf
import tensorflow.python.keras as k

#from tensorflow.python.keras.layers.experimental import preprocessing
from tensorflow.python.keras import layers
from tensorflow.python.keras import models

from keras.models import Sequential
from keras.engine.input_layer import Input
from keras.layers import Dense, TimeDistributed, Conv1D, Conv2D, Conv3D, ConvLSTM1D, ConvLSTM2D, MaxPooling1D, Convolution1D, Conv1DTranspose
from tensorflow.python.keras.layers.recurrent import GRU, LSTM
from tensorflow.python.platform.tf_logging import fatal

import tensorflow_io as tfio
from IPython.display import Audio
#from FTHelper import STFT, STFTInverse, STFTInverseShared
#from MyFFTLayer import mySTFT, mySTFTInverse

#from kapre import STFT, InverseSTFT, Magnitude, MagnitudeToDecibel
#from kapre.composed import get_melspectrogram_layer, get_log_frequency_spectrogram_layer

import librosa
import pydub
import scipy.io.wavfile

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

def write_audiofile_scipy(data, filepath):
    int_data = data[0:]
    length = len(int_data)
    assert(length > 0)
    #int_data = int_data * 32767
    #int_data = int_data.astype(np.int16)
    print(f"Writing file '{filepath}'")
    scipy.io.wavfile.write(filepath, 44100, int_data)

def read_audiofile_scipy(filepath):
    print(f"Reading file '{filepath}'")
    sanplerate, data = scipy.io.wavfile.read(filepath)
    assert sanplerate == 44100

    if data.dtype == np.int16:
        data = data.astype(np.float32)
        data = data * (1.0 / 32767.0)

    return data    
    

def read_audiofile(filepath):
    filename, file_extension = os.path.splitext(filepath)
    if file_extension == ".mp3":
        return read_audiofile_pydub(filepath)
    else:
        return read_audiofile_scipy(filepath)

    '''
    audio = tfio.audio.AudioIOTensor(filepath)   
    #samplerate = audio.rate.numpy()
    input_shape = audio.shape.numpy()
    bit_depth = audio.dtype
    #print (input_shape)
    audio_tensor = tf.cast(audio.to_tensor(), tf.float32)   
    audio_tensor = tf.scalar_mul(1.0 / (bit_depth.max + 1), audio_tensor)   #32768
    
    #factor = (bit_depth.max + 1) * 2.0
    #audio_tensor = tf.scalar_mul(1.0 / factor, audio_tensor)   #32768
    #audio_tensor = tf.add(audio_tensor, 0.5)
    #print(audio_tensor[100000:100100].numpy())
    return audio_tensor
    '''

def read_audiofile_pydub(filepath):
    a = pydub.AudioSegment.from_mp3(filepath)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    audio_tensor = y.astype('float32')
    #audio_tensor = np.scalar_mul(1.0 / 32768, y)   #32768    
    audio_tensor = audio_tensor * (1.0 / 32768)
    return audio_tensor


def build_model():
    use_bias = False

    model1 = Sequential()
    model1.add(Input(batch_shape=(cfg.BatchSize, cfg.InputSize, cfg.NumChannels)))
    
    #model1.add(Conv1DTranspose(HiddenSize, 9, padding='same', activation='linear', use_bias=use_bias))
    #model1.add(Conv1DTranspose(HiddenSize, 5, padding='same', activation='linear', use_bias=use_bias))
    #model1.add(Conv1DTranspose(HiddenSize, 3, padding='same', activation='linear', use_bias=use_bias))
    #model1.add(Convolution1D(HiddenSize, 16, padding='same', activation='linear', use_bias=use_bias))

    model1.add(GRU(cfg.HiddenSize, return_sequences=True, activation='linear', use_bias=use_bias, stateful=True))
    #model1.add(LSTM(cfg.HiddenSize, return_sequences=True, activation='linear', use_bias=use_bias, stateful=True))
    #model1.add(LSTM(cfg.HiddenSize, return_sequences=True, activation='linear', use_bias=use_bias, stateful=True))
    
    #model1.add(Convolution1D(HiddenSize, 4, padding='same', activation='linear', use_bias=use_bias))
    #model1.add(MaxPooling1D(pool_size=1, padding='same'))

    #model1.add(GRU(HiddenSize, return_sequences=True, activation='linear', use_bias=use_bias, stateful=True))
    #model1.add(LSTM(HiddenSize, return_sequences=True, activation='linear', use_bias=use_bias, stateful=True))
    #model1.add(TimeDistributed(Dense(NumChannels, activation='linear', use_bias=True)))
    #model1.add(TimeDistributed(Dense(NumChannels * 4, activation='linear', use_bias=use_bias)))
    #model1.add(TimeDistributed(Dense(NumChannels * 2, activation='linear', use_bias=use_bias)))
    model1.add(TimeDistributed(Dense(cfg.NumChannels, activation='linear', use_bias=use_bias)))    
    return model1

def build_model_v2():
    use_bias = True

    model1 = Sequential()
    model1.add(Input(batch_shape=(cfg.BatchSize, cfg.InputSize, cfg.NumChannels)))
        
    model1.add(Convolution1D(cfg.HiddenSize, kernel_size=3, dilation_rate=2, strides=1, groups=2, padding='causal', activation='linear', use_bias=use_bias))
    model1.add(Convolution1D(cfg.HiddenSize, kernel_size=3, dilation_rate=1, strides=1, groups=2, padding='causal', activation='linear', use_bias=use_bias))
    model1.add(Convolution1D(cfg.HiddenSize, kernel_size=1, dilation_rate=1, strides=1, groups=2, padding='causal', activation='linear', use_bias=use_bias))

    #model1.add(GRU(cfg.HiddenSize, return_sequences=True, activation='linear', use_bias=use_bias, stateful=True))    
    model1.add(LSTM(cfg.HiddenSize, return_sequences=True, activation='linear',recurrent_activation='linear', use_bias=use_bias, unit_forget_bias=False, stateful=True))
    model1.add(TimeDistributed(Dense(cfg.NumChannels, activation='linear', use_bias=use_bias)))    
    return model1

def build_model_v3():
    use_bias = True

    model1 = Sequential()
    model1.add(Input(batch_shape=(cfg.BatchSize, cfg.InputSize, cfg.NumChannels)))

    for i in range(1,7):
        model1.add(Convolution1D(cfg.HiddenSize, kernel_size=3, dilation_rate=i*2, strides=1, groups=2, padding='causal', activation='linear', use_bias=use_bias))

    #model1.add(GRU(cfg.HiddenSize, return_sequences=True, activation='linear', use_bias=use_bias, stateful=True))    
    model1.add(LSTM(cfg.HiddenSize, return_sequences=True, activation='linear',recurrent_activation='linear', use_bias=use_bias, unit_forget_bias=False, stateful=True))
    model1.add(TimeDistributed(Dense(cfg.NumChannels, activation='linear', use_bias=use_bias)))    
    return model1

def build_model_fft():
    frame_length = 63
    frame_step = 16
    n_freqs=150
    fft_trainable = True
    use_bias = False

    model1 = Sequential()
    #model1.add(Input(shape=(InputSize, NumChannels)))
    #model1.add(Input(shape=(InputSize)))
    model1.add(Input(batch_shape=(cfg.BatchSize, cfg.InputSize)))
    
    stft_layer = STFT(frame_length=frame_length, frame_step=frame_step, n_freqs=n_freqs, out_module=False, frequency_trainable=fft_trainable, window_trainable=fft_trainable)
    model1.add(stft_layer)    

    model1.add(TimeDistributed(GRU(cfg.HiddenSize, return_sequences=True, activation='linear', use_bias=use_bias)))
    model1.add(TimeDistributed(GRU(cfg.HiddenSize, return_sequences=True, activation='linear', use_bias=use_bias)))
    #model1.add(TimeDistributed(LSTM(cfg.HiddenSize, return_sequences=True, activation='linear', use_bias=use_bias, stateful=True)))    
    model1.add(TimeDistributed(Dense(cfg.NumChannels, activation='linear', use_bias=use_bias)))
    
    model1.add(STFTInverseShared(stft_layer))
    return model1

def build_model_fft_cnn():
    frame_length = 64
    frame_step = 16
    n_freqs=256
    fft_trainable = True
    use_bias = False

    model1 = Sequential()
    #model1.add(Input(shape=(InputSize, NumChannels)))
    model1.add(Input(shape=(cfg.InputSize)))

    stft_layer = STFT(frame_length=frame_length, 
        frame_step=frame_step, 
        n_freqs=n_freqs, 
        out_module=False, 
        frequency_trainable=fft_trainable, 
        window_trainable=fft_trainable)
    model1.add(stft_layer)
    
    #model1.add(TimeDistributed(GRU(HiddenSize, return_sequences=True, activation='linear', use_bias=use_bias)))
    model1.add(TimeDistributed(GRU(n_freqs, return_sequences=True, activation='linear', use_bias=use_bias)))
    #model1.add(TimeDistributed(GRU(n_freqs, return_sequences=True, activation='tanh', use_bias=use_bias)))

    #model1.add(ConvLSTM1D(64, 32, return_sequences=True, padding='same'))
    #model1.add(ConvLSTM1D(32, 16, return_sequences=True, padding='same'))
    #model1.add(ConvLSTM1D(128, 2, return_sequences=True, padding='same'))
    #model1.add(ConvLSTM1D(64, 1, return_sequences=True, padding='same', activation='linear'))

    model1.add(TimeDistributed(Dense(n_freqs, activation='linear', use_bias=use_bias)))
    model1.add(STFTInverseShared(stft_layer))
    return model1

def build_model_myfft():
    frame_length = 255
    frame_step = 128 
    use_bias = False

    model1 = Sequential()    
    model1.add(Input(shape=(cfg.InputSize)))
    model1.add(mySTFT(frame_length=frame_length, frame_step=frame_step))
    #model1.add(TimeDistributed(GRU(HiddenSize, return_sequences=True, activation='linear', use_bias=use_bias)))
    #model1.add(TimeDistributed(LSTM(HiddenSize, return_sequences=True, activation='linear', use_bias=use_bias)))
    #model1.add(LSTM(HiddenSize, return_sequences=True, dtype=tf.complex64, activation='linear', use_bias=use_bias))
    
    #model1.add(TimeDistributed(Dense(NumChannels, activation='linear', use_bias=use_bias)))    
    #model1.add(TimeDistributed(Dense(NumChannels, activation='linear', use_bias=use_bias)))
    model1.add(mySTFTInverse(frame_length=frame_length, frame_step=frame_step))
    return model1    

def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.LearningRate)        
    model.compile(optimizer=optimizer, loss='mae', metrics=['accuracy'])
    return model

def load_weights(model):
    if os.path.exists(cfg.ModelFileName):
        print(f"Loading model weights from '{cfg.ModelFileName}'")
        model.load_weights(cfg.ModelFileName)
