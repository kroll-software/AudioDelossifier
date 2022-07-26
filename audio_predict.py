import os
import pathlib
import keras
import numpy as np
import glob
from numpy.core.arrayprint import dtype_is_implied
import gc

import tensorflow as tf
import tensorflow.python.keras as k

import tensorflow_io as tfio
from IPython.display import Audio

import audio_functions as af
import audio_config as cfg

AUDIOPATH = "./audio/"
OUTPATH = AUDIOPATH + "out/"

model = af.build_model_v2()
model = af.compile_model(model)
model.summary()

af.load_weights(model)

def ProcessFile(fname=""):    
    print(f"Reading audio file '{fname}'")
    data = af.read_audiofile(fname)
    sl = len(data) // cfg.InputSize * cfg.InputSize
    data = np.stack(np.split(data[:sl], cfg.InputSize), axis=1)

    data_len = len(data) // cfg.BatchSize * cfg.BatchSize
    data = data[0:data_len]
    
    # process data in chunks
    Y_pred = []
    i, chunksize = 0, cfg.BatchSize * 1024
    for idx in range(0, len(data), chunksize):        
        Y_pred += list(model.predict(data[idx:(i+1)*chunksize], verbose=1, batch_size=cfg.BatchSize))
        i += 1
    Y_pred = np.array(Y_pred)    

    # collapse batch column, reduce dimension
    Y_pred = Y_pred.reshape((Y_pred.shape[0] * Y_pred.shape[1], Y_pred.shape[2]), order='F')

    out_fname, file_extension = os.path.splitext(fname)
    out_fname = f"{out_fname}-{cfg.Mp3BitRate}.wav"
    out_fname = out_fname.replace(AUDIOPATH, OUTPATH)
    af.write_audiofile_scipy(Y_pred, out_fname)

files = glob.glob(AUDIOPATH + "*.wav")
for file in files:    
    ProcessFile(file)

print ("Ready.")
