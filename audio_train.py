import os
import random
import numpy as np
import glob
from numpy.core.arrayprint import dtype_is_implied
from pydub import AudioSegment
from tqdm import tqdm

import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from mutagen.mp3 import MP3

import audio_functions as af
import audio_config as cfg
import audio_models as models
#import audio_models_experimental as experiments


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


def preprocess_dataset(align: bool=False):
    uncompressed_path = os.path.join(cfg.DATAPATH, "uncompressed")
    compressed_path = os.path.join(cfg.DATAPATH, "compressed")

    compressed_files = glob.glob(os.path.join(compressed_path, "*.mp3"))    
    random.shuffle(compressed_files)    

    x_train = np.zeros([0, cfg.InputSize, cfg.NumChannels], dtype=np.float32)
    y_train = np.zeros([0, cfg.InputSize, cfg.NumChannels], dtype=np.float32)        

    for f in compressed_files:        
        ok = False
        m_file = os.path.join(uncompressed_path, os.path.basename(f))
        m_file = os.path.splitext(m_file)[0] + '.wav'
        if not os.path.exists(m_file):
            m_file = os.path.splitext(m_file)[0] + '.flac'
                        
        if not os.path.exists(m_file):
            print(f"Uncompressed file '{m_file}' not found")
        else:
            if align:
                aligned_opath = f.replace("/training-data/", "/aligned-data/")
                aligned_mpath = m_file.replace("/training-data/", "/aligned-data/")
                
                aligned_opath = os.path.splitext(aligned_opath)[0] + '.wav'
                aligned_mpath = os.path.splitext(aligned_mpath)[0] + '.wav'
            else:
                aligned_opath = f
                aligned_mpath = m_file

            if os.path.exists(aligned_opath) and os.path.exists(aligned_mpath):                
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
                    af.write_audiofile_wav(odata, aligned_opath)                    
                    af.write_audiofile_wav(mdata, aligned_mpath)
                    ok = True
                else:
                    print(f"Offset too large ({offset}), file not exported.")
                print("")
        
            if ok == True:
                sl1 = len(odata) // cfg.InputSize * cfg.InputSize
                sl2 = len(mdata) // cfg.InputSize * cfg.InputSize
                sl = min(sl1, sl2)
                assert sl > 0
                x_train = np.concatenate([x_train, np.stack(np.split(odata[:sl], cfg.InputSize), axis=1)], 0)
                y_train = np.concatenate([y_train, np.stack(np.split(mdata[:sl], cfg.InputSize), axis=1)], 0)
    return x_train, y_train

def convert_flac2mp3(flac_filepath, mp3filepath, bitrate: str = "256k"):
    s = AudioSegment.from_file(flac_filepath)
    s.export(mp3filepath, bitrate=bitrate, format="mp3")

def create_mp3_files(bitrate: str = "256k"):
    compressed_path = os.path.join(cfg.DATAPATH, "compressed")
    if not os.path.exists(compressed_path):
        os.mkdir(compressed_path)

    uncompressed_path = os.path.join(cfg.DATAPATH, "uncompressed")
    if not os.path.exists(uncompressed_path):
        os.mkdir(uncompressed_path)

    compressed_files = glob.glob(os.path.join(compressed_path, '*.mp3'))

    # delete compresed files that don't match the bitrate        
    for f in compressed_files:        
        try:
            mp3 = MP3(f)
            mp3_bitrate = str(int(mp3.info.bitrate / 1000)) + "k"
            if mp3_bitrate != bitrate:
                os.remove(f)
        except:
            os.remove(f)

        

    types = ('*.wav', '*.flac') # the tuple of file types    
    uncompressed_files = []    
    for extension in types:
        uncompressed_files.extend(glob.glob(os.path.join(uncompressed_path, extension)))
    uncompressed_files.sort()    

    for file in tqdm(uncompressed_files, desc="Converting audio-files to *.mp3"):
        targetpath = os.path.join(compressed_path, os.path.basename(file))        
        targetpath = os.path.splitext(targetpath)[0] + '.mp3'
        if not os.path.exists(targetpath):            
            convert_flac2mp3(file, targetpath, bitrate)


def get_file_paths():
    uncompressed_path = os.path.join(cfg.DATAPATH, "uncompressed")
    compressed_path = os.path.join(cfg.DATAPATH, "compressed")
    
    cp = glob.glob(os.path.join(compressed_path, "*.mp3"))
    
    compressed_files = []
    uncompressed_files = []
    for f in cp:
        m_file = os.path.join(uncompressed_path, os.path.basename(f))
        m_file = os.path.splitext(m_file)[0] + '.wav'
        if not os.path.exists(m_file):
            m_file = os.path.splitext(m_file)[0] + '.flac'
                        
        if os.path.exists(m_file):
            compressed_files.append(f)
            uncompressed_files.append(m_file)

    return compressed_files, uncompressed_files



def preprocess(input_path, label_path, input_len=512):    
    input_melspec = af.load_melspec(input_path, sr_out=44100, input_len=input_len)
    label_melspec = af.load_melspec(label_path, sr_out=44100, input_len=input_len)

    return input_melspec, label_melspec


def create_dataset_with_spectrogram_labels(
    input_file_paths, label_file_paths, batch_size, 
    target_sample_rate=44100, sr_out=8000, n_fft=256, hop_length=128, input_len=512
):
    def tf_preprocess(input_path, label_path):
        input_melspec, label_melspec = tf.numpy_function(
            preprocess, [input_path, label_path, input_len], [tf.float32, tf.float32]
        )
        # Set fixed shapes after processing
        input_melspec.set_shape([input_len, 128, 2])
        label_melspec.set_shape([input_len, 128, 2])
        return input_melspec, label_melspec

    dataset = tf.data.Dataset.from_tensor_slices((input_file_paths, label_file_paths))
    dataset = dataset.map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Shuffle the dataset if needed
    dataset = dataset.shuffle(buffer_size=len(input_file_paths))
    
    # Set drop_remainder=True to ensure all batches have exactly batch_size samples
    #dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    return dataset

def report_base_metrics(x_train, y_train):
    # report some base metrics
    try:
        # may throw OOM errors
        mae = tf.keras.metrics.MeanAbsoluteError()
        mae.update_state(x_train, y_train)
        print(f'Base mae of the data without training: {mae.result().numpy()}')
    except:
        mae = af.calc_mae(x_train, y_train)
        print(f'Base mae of the data without training: {mae}')

    try:
        mse = tf.keras.metrics.MeanSquaredError()
        mse.update_state(x_train, y_train)
        print(f'Base mse of the data without training: {mse.result().numpy()}')
    except:
        mse = af.calc_mse(x_train, y_train)
        print(f'Base mse of the data without training: {mse}')

    try:
        acc = tf.keras.metrics.Accuracy()
        acc.update_state(x_train, y_train)
        print(f'Base accuracy of the data without training: {acc.result().numpy()}')
    except:
        acc = af.calc_accuracy(x_train, y_train)
        print(f'Base accuracy of the data without training: {acc}')


if __name__ == "__main__":        
    
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    tf.config.run_functions_eagerly(True)

    create_mp3_files(bitrate=cfg.Mp3BitRate)

    x_train, y_train = preprocess_dataset()
    
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(buffer_size=len(x_train))
        .batch(cfg.BatchSize, drop_remainder=True)
        #.repeat()  # Ensure the dataset repeats continuously
        .prefetch(tf.data.AUTOTUNE)  # Optimize for pipeline efficiency        
    )        

    model = models.build_model()    
    
    # test the model
    first_batch = next(iter(train_dataset))
    input_shape = first_batch[0].shape
    print(f"batch input shape: {input_shape}")
    
    # call model with first batch to debug
    output = model(first_batch[0])
    output_shape = output.shape    
    print(f"model output shape: {output_shape}")    

    # final test for consistence
    assert input_shape == output_shape

    report_base_metrics(x_train, y_train)
    
    # load weights from previous trainings
    model_save_path = cfg.ModelFileName()
    if os.path.exists(model_save_path):
        print(f"Loading model weights from '{model_save_path}'")
        model.load_weights(model_save_path)

    # model checkpoint callback to save weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
        monitor="loss", mode="min",
        save_best_only=True,
        save_weights_only=True,
        verbose=0,
    )

    # log to tensorboard callback
    #tensorboard_cb = tf.keras.callbacks.TensorBoard()

    # learning-rate scheduler callback
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', mode= 'min',
        factor=0.2,
        min_lr=1e-5,
        patience=3,
        cooldown=2,
        verbose=1,
        )    

    # training loop    
    history = model.fit(train_dataset,     
        batch_size=cfg.BatchSize, 
        epochs=cfg.Epochs,      
        callbacks=[cp_callback, lr_schedule], 
        shuffle=True)

    print(f"Training Complete: Accuracy {history.history['accuracy'][-1]}")  