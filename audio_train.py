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
from keras.layers import GRU, LSTM, Dense, TimeDistributed, Conv1D, Conv2D, Conv3D, ConvLSTM1D, ConvLSTM2D, MaxPooling1D, Convolution1D, Conv1DTranspose
#from keras.layers.recurrent import GRU, LSTM
from tensorflow.python.platform.tf_logging import fatal

import tensorflow_io as tfio
from IPython.display import Audio
#from FTHelper import STFT, STFTInverse, STFTInverseShared
#from MyFFTLayer import mySTFT, mySTFTInverse
from sklearn.metrics import mean_absolute_error

import audio_functions as af
import audio_config as cfg

def auto_align_samples(data1, data2, num_samples=44100, center_percent=0.5):
    center = int(len(data1) * center_percent)
    offset = 0
    best_diff = float("inf")
    # search from center to the left    
    for i in range(0, -num_samples // 2, -1):
        td1 = data1[center: center + num_samples]
        td2 = data2[center + i: center + i + num_samples]        
        #difference = np.sum(tf.keras.metrics.mean_absolute_error(td1, td2).numpy())
        difference = mean_absolute_error(td1, td2)
        if difference < best_diff:
            best_diff = difference
            offset = i    

    # search from center to the right
    for i in range(0, num_samples // 2):
        td1 = data1[center: center + num_samples]
        td2 = data2[center + i: center + i + num_samples]        
        #difference = np.sum(tf.keras.metrics.mean_absolute_error(td1, td2).numpy())    
        difference = mean_absolute_error(td1, td2)
        if difference < best_diff:
            best_diff = difference
            offset = i
    
    print(f"Auto-Align Waves, Offset: {offset}, Difference: {best_diff}")    
    
    min_len = min(len(data1), len(data2)) - abs(offset)

    if offset < 0:        
        td1 = data1[-offset:min_len-offset]
        td2 = data2[0:min_len]
    else:        
        td1 = data1[0:min_len]
        td2 = data2[offset:min_len+offset]
    
    assert(len(td1) == len(td2))    
    return td1, td2, offset


def preprocess_dataset():
    compressed_files = glob.glob(cfg.DATAPATH + "compressed/*.wav")
    #uncompressed_files = glob.glob(cfg.DATAPATH + "uncompressed/*.wav")    

    x_train = np.zeros([0, cfg.InputSize, cfg.NumChannels], dtype=np.float32)
    y_train = np.zeros([0, cfg.InputSize, cfg.NumChannels], dtype=np.float32)        

    for f in compressed_files:        
        ok = False                
        m_file = f.replace("/compressed/", "/uncompressed/")
                        
        if not pathlib.Path(m_file).is_file():
            print(f"Uncompressed file '{m_file}' not found")
        else:
            aligned_opath = f.replace("/training-data/", "/aligned-data/")
            aligned_mpath = m_file.replace("/training-data/", "/aligned-data/")            

            if pathlib.Path(aligned_opath).is_file() and pathlib.Path(aligned_mpath).is_file():
                print(f"Loading aligned audio-files '{aligned_opath}'")
                odata = af.read_audiofile(aligned_opath)
                mdata = af.read_audiofile(aligned_mpath)
                assert(len(odata) == len(mdata))
                ok = True
            else:
                odata = af.read_audiofile(f)
                mdata = af.read_audiofile(m_file)

                delta_len = abs(len(odata) - len(mdata)) * 2
                num_samples = int(44100 + delta_len)
                
                print(f"Processing {m_file}")                                
                print(f"Align Step 1, testing {num_samples} samples")
                odata, mdata, offset = auto_align_samples(odata, mdata, num_samples, 0.25)
                print("Align Step 2")
                odata, mdata, offset = auto_align_samples(odata, mdata, num_samples, 0.75)                
                
                if abs(offset) < 1:                    
                    af.write_audiofile_scipy(odata, aligned_opath)
                    af.write_audiofile_scipy(mdata, aligned_mpath)                    
                    ok = True
                else:
                    print(f"Offset too large ({offset}), file not exported.")
                print("")
        
            if ok == True:
                sl = len(odata) // cfg.InputSize * cfg.InputSize                
                x_train = np.concatenate([x_train, np.stack(np.split(odata[:sl], cfg.InputSize), axis=1)], 0)
                y_train = np.concatenate([y_train, np.stack(np.split(mdata[:sl], cfg.InputSize), axis=1)], 0)                        
    return x_train, y_train

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cfg.ModelFileName,
    monitor="accuracy", mode="max",    
    save_best_only=True,
    save_weights_only=True,    
    verbose=1,
)

#model = af.build_model()
model = af.build_model_v2()
#model = af.build_model_v3()

#model = af.build_model_fft()
#model = af.build_model_fft_cnn()
#model = af.build_model_myfft()
model = af.compile_model(model)
model.summary()

af.load_weights(model)

x_train, y_train = preprocess_dataset()
print (f"Traindata length: {len(x_train)}")

train_len = len(x_train) // cfg.BatchSize * cfg.BatchSize
x_train = x_train[0:train_len]
y_train = y_train[0:train_len]

# TEST: convert to mono, left channel
if False:
    x_train = np.delete(x_train, 1, 2)
    y_train = np.delete(y_train, 1, 2)
    x_train = np.squeeze(x_train, 2)
    y_train = np.squeeze(y_train, 2)

acc = tf.keras.metrics.Accuracy()
acc.update_state(x_train, y_train)
print(f'Base accuracy of the data without training: {acc.result().numpy()}')

mae = tf.keras.metrics.MeanAbsoluteError()
mae.update_state(x_train, y_train)
print(f'Base mae of the data without training: {mae.result().numpy()}')

mse = tf.keras.metrics.MeanSquaredError()
mse.update_state(x_train, y_train)
print(f'Base mse of the data without training: {mse.result().numpy()}')

model.fit(x_train, y_train, batch_size=cfg.BatchSize, epochs=cfg.Epochs, callbacks=[cp_callback], shuffle=False)