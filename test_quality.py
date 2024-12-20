import os
import numpy as np
from pydub import AudioSegment
import soundfile as sf

import audio_functions as af
import audio_predict as ap
import audio_models as models

import audio_config as cfg

def read_wav(file_path):
    data, sample_rate = sf.read(file_path)    
    return data, sample_rate

def read_mp3(file_path):
    audio = AudioSegment.from_mp3(file_path)
    data = np.array(audio.get_array_of_samples())
    return data, audio.frame_rate

def read_flac(file_path):
    data, sample_rate = sf.read(file_path)
    return data, sample_rate

def normalize(signal):
    return signal / np.max(np.abs(signal))

def convert_flac2mp3(flac_filepath, mp3filepath, bitrate="128k"):
    s = AudioSegment.from_file(flac_filepath)
    s.export(mp3filepath, bitrate=bitrate, format="mp3")


if __name__ == "__main__":

    # copy and name a uncompressed file for testing    
    org_file_path = './tests/Carl Orff - O fortuna.flac'
    mp3_file_path = os.path.splitext(org_file_path)[0] + ".mp3"
    
    convert_flac2mp3(org_file_path, mp3_file_path, cfg.Mp3BitRate)

    org_data = af.read_audiofile(org_file_path)
    mp3_data = af.read_audiofile(mp3_file_path)

    models.load_weights(ap.model, cfg.Mp3BitRate)
    delossified_data = ap.ProcessFileData(mp3_data)

    l1 = len (org_data)
    l2 = len (mp3_data)
    assert l1 == l2

    delossified_data = delossified_data[0:l1]

    l3 = len(delossified_data)
    assert l3 == l1

    dtype1 = org_data.dtype
    dtype2 = mp3_data.dtype
    dtype3 = delossified_data.dtype
    assert dtype1 == dtype2 == dtype3

    min1 = org_data.min()
    min2 = mp3_data.min()
    min3 = delossified_data.min()

    max1 = org_data.max()
    max2 = mp3_data.max()
    max3 = delossified_data.max()
    
    mse1 = af.calc_mse(org_data, mp3_data)
    mse2 = af.calc_mse(org_data, delossified_data)

    print(f"MSE org_data, mp3_data: {mse1}")
    print(f"MSE org_data, delossified_data: {mse2}")

    if mse2 < mse1:
        print(f"SUCCESS: mp3_data was improved by {mse1 - mse2}")
    else:
        print(f"Sorry: delossified is worse by {mse2 - mse1}")    

    print("Ready.")