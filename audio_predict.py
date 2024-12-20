import glob
import os
import numpy as np

#import analysis as an
from mutagen.mp3 import MP3

import audio_config as cfg
import audio_functions as af
import audio_models as models

AUDIOPATH = "./audio/"
OUTPATH = AUDIOPATH + "out/"

# check GPU-support
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model = models.build_model()

#af.load_weights(model)
last_bitrate = None

def ProcessFileData(data):
    # Pad data for batch-processing, fill with zeros at the end
    data_len = len(data)
    total_padsize = cfg.InputSize * cfg.BatchSize
    if (data_len % total_padsize > 0):
        data_len = len(data) // total_padsize * total_padsize + total_padsize
    pad_size = data_len - len(data)
    
    zeros = np.zeros((pad_size, data.shape[1]), dtype=data.dtype, order="C")
    data = np.append(data, zeros, axis=0)

    data = np.stack(np.split(data, cfg.InputSize), axis=1)    
    
    # process data in chunks
    Y_pred = []
    i, chunksize = 0, cfg.BatchSize * 1024    
    for idx in range(0, len(data), chunksize):        
        Y_pred += list(model.predict(data[idx:(i+1)*chunksize], verbose=1, batch_size=cfg.BatchSize))
        #Y_pred += list(data[idx:(i+1)*chunksize])  #for testing without ML
        i += 1
    Y_pred = np.array(Y_pred)    

    # collapse batch column, reduce dimension
    Y_pred = Y_pred.reshape((Y_pred.shape[0] * Y_pred.shape[1], Y_pred.shape[2]), order='F')
    return Y_pred

def ProcessFile(fname: str) -> str:
    global last_bitrate
    bitrate = cfg.Mp3BitRate
    if os.path.splitext(fname)[1] == ".mp3":
        f = MP3(fname)
        bitrate = str(int(f.info.bitrate / 1000)) + "k"

    if bitrate != last_bitrate:
        models.load_weights(model, bitrate)
        last_bitrate = bitrate

    data = af.read_audiofile(fname)
    data_out = ProcessFileData(data)
    
    #mse = af.calc_mse(data, data_out)
    #print(f"MSE: {mse}")    

    
    outputformat = cfg.OUTPUTFORMAT    
    if outputformat and outputformat.lower().find("flac") > -1:
        out_fname = os.path.basename(fname)
        out_fname = os.path.splitext(out_fname)[0] + ".flac"
        out_fname = os.path.join(OUTPATH, out_fname)
        af.write_audiofile_flac(data_out, out_fname)
    else:
        out_fname = os.path.basename(fname)
        out_fname = os.path.splitext(out_fname)[0] 
        out_fname = f"{out_fname}-{bitrate}.wav"
        out_fname = os.path.join(OUTPATH, out_fname)
        af.write_audiofile_wav(data_out, out_fname)
    
    return out_fname

if __name__ == "__main__":
    types = ('*.wav', '*.mp3') # the tuple of file types
    files = []
    for extension in types:
        files.extend(glob.glob(os.path.join(AUDIOPATH, extension)))
    files.sort()
    for file in files:    
        ProcessFile(file)

    print ("Ready.")
