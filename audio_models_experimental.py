import os
import numpy as np
import tensorflow as tf
# VS-Code Python debugger can't resolve this, but should still work
#from tensorflow.keras import layers
# instead you could 'pip install keras' and use
from keras import layers
import librosa
import audio_config as cfg

def build_model_v2_functional():
    use_bias = True
    
    inputs = layers.Input(batch_shape=(cfg.BatchSize, cfg.InputSize, cfg.NumChannels))    

    x = layers.Conv1D(cfg.HiddenSize, kernel_size=3, dilation_rate=2, strides=1, groups=2, padding='causal', activation='linear', use_bias=use_bias)(inputs)    
    x = layers.Conv1D(cfg.HiddenSize, kernel_size=3, dilation_rate=1, strides=1, groups=2, padding='causal', activation='linear', use_bias=use_bias)(x)
    x = layers.Conv1D(cfg.HiddenSize, kernel_size=1, dilation_rate=1, strides=1, groups=2, padding='causal', activation='linear', use_bias=use_bias)(x)
    x = layers.LSTM(cfg.HiddenSize, return_sequences=True, activation='linear',recurrent_activation='linear', use_bias=use_bias, unit_forget_bias=False, stateful=True)(x)    
    outputs = layers.TimeDistributed(layers.Dense(cfg.NumChannels, activation='linear', use_bias=use_bias))(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)    


def build_audio_delossifier(input_shape):
    # Encoder
    inputs = layers.Input(shape=input_shape)  # input_shape = (2048, 2)
    x = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    
    # Bottleneck
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense((input_shape[0] // 4) * 128, activation='relu')(x)  # Match flattened size
    x = layers.Reshape((input_shape[0] // 4, 128))(x)
    
    # Decoder
    x = layers.Conv1DTranspose(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1DTranspose(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(size=2)(x)
    
    # Output Layer
    outputs = layers.Conv1D(2, kernel_size=3, activation='linear', padding='same')(x)  # 2 channels for stereo
    
    return tf.keras.Model(inputs, outputs)


def load_melspec(file_path, target_sample_rate=44100, sr_out=8000, n_fft=256, hop_length=128, input_len=512):
    """
    Load an audio file and convert it to a fixed-length Mel-Spectrogram.

    Args:
        file_path (str): Path to the audio file.
        target_sample_rate (int): Desired output sample rate.
        sr_out (int): Resampling rate for Mel Spectrogram output.
        n_fft (int): Number of FFT components.
        hop_length (int): Hop length parameter for STFT.
        input_len (int): Fixed length for slicing or padding.

    Returns:
        np.ndarray: Mel-Spectrogram with shape [input_len, num_mel_bins, 2].
    """
    y, sr = librosa.load(file_path, sr=target_sample_rate, mono=False)
    
    # Ensure stereo data
    if len(y.shape) == 1:
        y = np.tile(y[:, np.newaxis], (1, 2))  # Duplicate mono to stereo
    left_stereo, right_stereo = y[0], y[1]
    
    # Convert to Mel-Spectrogram for each channel
    mel_left = librosa.feature.melspectrogram(y=left_stereo, sr=sr_out, n_fft=n_fft, hop_length=hop_length)
    mel_right = librosa.feature.melspectrogram(y=right_stereo, sr=sr_out, n_fft=n_fft, hop_length=hop_length)

    # Ensure consistent length: slice or pad
    def pad_or_trim(mel, length):
        if mel.shape[1] > length:
            return mel[:, :length]
        else:
            return np.pad(mel, ((0, 0), (0, length - mel.shape[1])), mode='constant')

    mel_left = pad_or_trim(mel_left, input_len)
    mel_right = pad_or_trim(mel_right, input_len)

    # Stack and transpose to match shape [input_len, num_mel_bins, channels]
    melspecs = np.stack([mel_left, mel_right], axis=-1).transpose(1, 0, 2)
    
    return melspecs  # Shape: [input_len, num_mel_bins, 2]


class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=4, ff_dim=None, rate=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if ff_dim is None:
            self.ff_dim = 4 * embed_dim
        else:
            self.ff_dim = ff_dim
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_ln = layers.LayerNormalization(epsilon=1e-6)
        
        self.fc1 = layers.Dense(self.ff_dim, activation="relu")
        self.fc2 = layers.Dense(embed_dim)
        self.dense_norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Split inputs into query, key, and value for self-attention
        attn_output = self.attn(
            query=inputs,  # Query is the input tensor
            key=inputs,    # Key is also the input tensor (self-attention)
            value=inputs,  # Value is the input tensor
            attention_mask=None,
            return_attention_scores=False
        )
        
        # Add and normalize
        x1 = self.attention_ln(attn_output + inputs)
        
        # Feed-forward network
        f1 = self.fc1(x1)
        f2 = self.fc2(f1)
        x2 = self.dense_norm(f2 + x1)
        
        return x2


    def compute_output_shape(self, input_shape):
        return input_shape
    

def multichannel_unet(input_shape=(None, 128, 2)) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
        
    # Contracting Path (Separate for each channel)
    c1_left = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs[:, :, :, 0])
    attn_c1_left = SelfAttentionBlock(64)(c1_left)    
    p1_left = layers.MaxPooling1D((2,))(attn_c1_left)

    c2_left = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(p1_left)
    attn_c2_left = SelfAttentionBlock(128)(c2_left)
    p2_left = layers.MaxPooling1D((2))(attn_c2_left)

    c3_left = layers.Conv1D(256, kernel_size=5, activation='relu', padding='same')(p2_left)
    attn_c3_left = SelfAttentionBlock(256)(c3_left)


    # Contracting Path (Separate for each channel)
    c1_right = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs[:, :, :, 1])
    attn_c1_right = SelfAttentionBlock(64)(c1_right)
    p1_right = layers.MaxPooling1D((2,))(attn_c1_right)

    c2_right = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(p1_right)
    attn_c2_right = SelfAttentionBlock(128)(c2_right)
    p2_right = layers.MaxPooling1D((2))(attn_c2_right)

    c3_right = layers.Conv1D(256, kernel_size=5, activation='relu', padding='same')(p2_right)
    attn_c3_right = SelfAttentionBlock(256)(c3_right)


    # BottleNeck (Concatenate both channels before bottlenecks)
    concat_bneck = layers.Concatenate()([attn_c3_left, attn_c3_right])
    
    x_bottleneck = layers.Conv1D(512, kernel_size=5, activation='relu', padding='same')(concat_bneck)

    # Expansive Path (Reverse operations with skip connections)
    u1_left = layers.UpSampling1D((2))(x_bottleneck[:, :, :256])
    conc_cat1_left = layers.Concatenate()([u1_left, attn_c2_left])
    
    x1r_left = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(conc_cat1_left)

    u2_left = layers.UpSampling1D((2))(x1r_left)
    conc_cat2_left = layers.Concatenate()([u2_left, attn_c1_left])
    
    x2r_left = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(conc_cat2_left)
    
    # Left Channel Output
    output_left = layers.Conv1D(2, kernel_size=3, activation='sigmoid', padding='same', name="output_left")(x2r_left)

    ## Expansive Path (Right Channel)
    u1_right = layers.UpSampling1D((2))(x_bottleneck[:, :, 256:])
    conc_cat1_right = layers.Concatenate()([u1_right, attn_c2_right])
    
    x1r_right = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(conc_cat1_right)

    u2_right = layers.UpSampling1D((2))(x1r_right)
    conc_cat2_right = tf.keras.layers.Concatenate()([u2_right, attn_c1_right])
    
    x2r_right = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(conc_cat2_right)
    
    # Right Channel Output
    output_right = layers.Conv1D(2, kernel_size=3, activation='sigmoid', padding='same', name="output_right")(x2r_right)

    total_output = layers.Concatenate(axis=0)([output_left, output_right])
    model = tf.keras.Model(inputs=[inputs], outputs=total_output)    

    model = tf.keras.Model(inputs=[inputs], outputs=total_output)
    #model = tf.keras.Model(inputs=[inputs], outputs={'left': output_left, 'right': output_right})

    return model


##########################################

def multichannel_unet_fixed(input_shape=(None, 128, 2)) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
        
    # Contracting Path (Separate for each channel)
    c1_left = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs[:, :, :, 0])
    attn_c1_left = SelfAttentionBlock(64)(c1_left)    
    p1_left = layers.MaxPooling1D((2,))(attn_c1_left)

    c2_left = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(p1_left)
    attn_c2_left = SelfAttentionBlock(128)(c2_left)
    p2_left = layers.MaxPooling1D((2))(attn_c2_left)

    c3_left = layers.Conv1D(256, kernel_size=5, activation='relu', padding='same')(p2_left)
    attn_c3_left = SelfAttentionBlock(256)(c3_left)


    # Contracting Path (Separate for each channel)
    c1_right = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs[:, :, :, 1])
    attn_c1_right = SelfAttentionBlock(64)(c1_right)
    p1_right = layers.MaxPooling1D((2,))(attn_c1_right)

    c2_right = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(p1_right)
    attn_c2_right = SelfAttentionBlock(128)(c2_right)
    p2_right = layers.MaxPooling1D((2))(attn_c2_right)

    c3_right = layers.Conv1D(256, kernel_size=5, activation='relu', padding='same')(p2_right)
    attn_c3_right = SelfAttentionBlock(256)(c3_right)


    # BottleNeck (Concatenate both channels before bottlenecks)
    concat_bneck = layers.Concatenate()([attn_c3_left, attn_c3_right])
    
    x_bottleneck = layers.Conv1D(512, kernel_size=5, activation='relu', padding='same')(concat_bneck)

    # Expansive Path (Reverse operations with skip connections)
    u1_left = layers.UpSampling1D((2))(x_bottleneck[:, :, :256])
    conc_cat1_left = layers.Concatenate()([u1_left, attn_c2_left])
    
    x1r_left = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(conc_cat1_left)

    u2_left = layers.UpSampling1D((2))(x1r_left)
    conc_cat2_left = layers.Concatenate()([u2_left, attn_c1_left])
    
    x2r_left = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(conc_cat2_left)
    
    # Left Channel Output
    output_left = layers.Conv1D(2, kernel_size=3, activation='sigmoid', padding='same', name="output_left")(x2r_left)

    ## Expansive Path (Right Channel)
    u1_right = layers.UpSampling1D((2))(x_bottleneck[:, :, 256:])
    conc_cat1_right = layers.Concatenate()([u1_right, attn_c2_right])
    
    x1r_right = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(conc_cat1_right)

    u2_right = layers.UpSampling1D((2))(x1r_right)
    conc_cat2_right = tf.keras.layers.Concatenate()([u2_right, attn_c1_right])
    
    x2r_right = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(conc_cat2_right)
    
    # Right Channel Output
    output_right = layers.Conv1D(2, kernel_size=3, activation='sigmoid', padding='same', name="output_right")(x2r_right)
    
    total_output = layers.Concatenate(axis=-1)([output_left, output_right])
    model = tf.keras.Model(inputs=[inputs], outputs=total_output)    
    
    return model


def multichannel_unet_fixed2(input_shape=(None, 128, 2)) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)

    # Split input into two channels
    left_channel = layers.Lambda(lambda x: x[..., 0:1])(inputs)  # Shape: (None, 512, 128, 1)
    right_channel = layers.Lambda(lambda x: x[..., 1:2])(inputs)  # Shape: (None, 512, 128, 1)

    # Contracting Path (Left Channel)
    c1_left = layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(left_channel)
    attn_c1_left = SelfAttentionBlock(64)(c1_left)
    p1_left = layers.MaxPooling2D((2, 2))(attn_c1_left)

    c2_left = layers.Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same')(p1_left)
    attn_c2_left = SelfAttentionBlock(128)(c2_left)
    p2_left = layers.MaxPooling2D((2, 2))(attn_c2_left)

    c3_left = layers.Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same')(p2_left)
    attn_c3_left = SelfAttentionBlock(256)(c3_left)

    # Contracting Path (Right Channel)
    c1_right = layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(right_channel)
    attn_c1_right = SelfAttentionBlock(64)(c1_right)
    p1_right = layers.MaxPooling2D((2, 2))(attn_c1_right)

    c2_right = layers.Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same')(p1_right)
    attn_c2_right = SelfAttentionBlock(128)(c2_right)
    p2_right = layers.MaxPooling2D((2, 2))(attn_c2_right)

    c3_right = layers.Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same')(p2_right)
    attn_c3_right = SelfAttentionBlock(256)(c3_right)

    # Bottleneck
    concat_bneck = layers.Concatenate()([attn_c3_left, attn_c3_right])
    x_bottleneck = layers.Conv2D(512, kernel_size=(5, 5), activation='relu', padding='same')(concat_bneck)

    # Expansive Path (Left Channel)
    u1_left = layers.UpSampling2D((2, 2))(x_bottleneck[..., :256])
    conc_cat1_left = layers.Concatenate()([u1_left, attn_c2_left])
    x1r_left = layers.Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same')(conc_cat1_left)

    u2_left = layers.UpSampling2D((2, 2))(x1r_left)
    conc_cat2_left = layers.Concatenate()([u2_left, attn_c1_left])
    x2r_left = layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(conc_cat2_left)

    # Left Channel Output
    output_left = layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(x2r_left)

    # Expansive Path (Right Channel)
    u1_right = layers.UpSampling2D((2, 2))(x_bottleneck[..., 256:])
    conc_cat1_right = layers.Concatenate()([u1_right, attn_c2_right])
    x1r_right = layers.Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same')(conc_cat1_right)

    u2_right = layers.UpSampling2D((2, 2))(x1r_right)
    conc_cat2_right = layers.Concatenate()([u2_right, attn_c1_right])
    x2r_right = layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(conc_cat2_right)

    # Right Channel Output
    output_right = layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(x2r_right)

    # Combine left and right channels
    combined_output = layers.Concatenate(axis=-1)([output_left, output_right])

    # Ensure final shape matches input shape
    final_output = layers.Reshape(input_shape)(combined_output)

    model = tf.keras.Model(inputs=inputs, outputs=final_output)

    return model


class LocalAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, kernel_size=(5, 5)):
        super(LocalAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            attention_axes=(1, 2)  # Restrict attention to local patches
        )
        self.conv = layers.Conv2D(
            filters=input_shape[-1],
            kernel_size=self.kernel_size,
            padding="same",
            activation="relu",
        )

    def call(self, inputs):
        # Local attention works within patches
        x = self.conv(inputs)
        x = self.attention(x, x)
        return x


def multichannel_unet_fixed3(input_shape=(None, 128, 2)) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)

    # Split input into two channels
    left_channel = layers.Lambda(lambda x: x[..., 0:1])(inputs)  # Shape: (None, 512, 128, 1)
    right_channel = layers.Lambda(lambda x: x[..., 1:2])(inputs)  # Shape: (None, 512, 128, 1)

    # Contracting Path (Left Channel)
    c1_left = layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(left_channel)
    attn_c1_left = LocalAttentionBlock(num_heads=4, key_dim=16)(c1_left)
    p1_left = layers.MaxPooling2D((2, 2))(attn_c1_left)

    c2_left = layers.Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same')(p1_left)
    attn_c2_left = LocalAttentionBlock(num_heads=4, key_dim=32)(c2_left)
    p2_left = layers.MaxPooling2D((2, 2))(attn_c2_left)

    c3_left = layers.Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same')(p2_left)
    attn_c3_left = LocalAttentionBlock(num_heads=4, key_dim=64)(c3_left)

    # Contracting Path (Right Channel)
    c1_right = layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(right_channel)
    attn_c1_right = LocalAttentionBlock(num_heads=4, key_dim=16)(c1_right)
    p1_right = layers.MaxPooling2D((2, 2))(attn_c1_right)

    c2_right = layers.Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same')(p1_right)
    attn_c2_right = LocalAttentionBlock(num_heads=4, key_dim=32)(c2_right)
    p2_right = layers.MaxPooling2D((2, 2))(attn_c2_right)

    c3_right = layers.Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same')(p2_right)
    attn_c3_right = LocalAttentionBlock(num_heads=4, key_dim=64)(c3_right)

    # Bottleneck
    concat_bneck = layers.Concatenate()([attn_c3_left, attn_c3_right])
    x_bottleneck = layers.Conv2D(512, kernel_size=(5, 5), activation='relu', padding='same')(concat_bneck)

    # Expansive Path (Left Channel)
    u1_left = layers.UpSampling2D((2, 2))(x_bottleneck[..., :256])
    conc_cat1_left = layers.Concatenate()([u1_left, attn_c2_left])
    x1r_left = layers.Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same')(conc_cat1_left)

    u2_left = layers.UpSampling2D((2, 2))(x1r_left)
    conc_cat2_left = layers.Concatenate()([u2_left, attn_c1_left])
    x2r_left = layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(conc_cat2_left)

    # Left Channel Output
    output_left = layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(x2r_left)

    # Expansive Path (Right Channel)
    u1_right = layers.UpSampling2D((2, 2))(x_bottleneck[..., 256:])
    conc_cat1_right = layers.Concatenate()([u1_right, attn_c2_right])
    x1r_right = layers.Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same')(conc_cat1_right)

    u2_right = layers.UpSampling2D((2, 2))(x1r_right)
    conc_cat2_right = layers.Concatenate()([u2_right, attn_c1_right])
    x2r_right = layers.Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(conc_cat2_right)

    # Right Channel Output
    output_right = layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(x2r_right)

    # Combine left and right channels
    combined_output = layers.Concatenate(axis=-1)([output_left, output_right])

    # Ensure final shape matches input shape
    final_output = layers.Reshape(input_shape)(combined_output)

    model = tf.keras.Model(inputs=inputs, outputs=final_output)

    return model

def perceptual_loss(y_true, y_pred):
    target_length = 2048    # guessed, don't know

    vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=(target_length, 2))
    vgg.trainable = False
    
    # Extract relevant layers (e.g., before the final pooling layer)
    outputs = [vgg.layers[i].output for i in range(7)]
    vgg_model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

    true_features = vgg_model(y_true)
    pred_features = vgg_model(y_pred)

    perceptual_loss_value = sum(tf.keras.backend.mean(tf.keras.backend.abs(t - p), axis=[1, 2, 3]) for t, p in zip(true_features, pred_features))    

    return perceptual_loss_value / len(outputs)


################ STFT Model #########################

# Helper Functions for STFT and ISTFT
def stft_layer(x, frame_length=512, frame_step=256):
    return tf.signal.stft(
        x, frame_length=frame_length, frame_step=frame_step, fft_length=frame_length,
        window_fn=tf.signal.hann_window
    )

def istft_layer(x, frame_length=512, frame_step=256):
    return tf.signal.inverse_stft(
        x, frame_length=frame_length, frame_step=frame_step, 
        window_fn=tf.signal.hann_window
    )

# Define the model
def build_fft_audio_model(input_shape=(2048, 2), fft_frame_size=512, fft_overlap=256, residual_channels=64, skip_channels=128):
    inputs = layers.Input(shape=input_shape)

    # **STFT Encoding**
    def stft_encoding(x):
        stft = tf.signal.stft(x[:, :, 0], frame_length=fft_frame_size, frame_step=fft_frame_size - fft_overlap, pad_end=True)
        stft_imag = tf.signal.stft(x[:, :, 1], frame_length=fft_frame_size, frame_step=fft_frame_size - fft_overlap, pad_end=True)
        stft_encoded = tf.complex(real=tf.math.real(stft), imag=tf.math.imag(stft_imag))
        return stft_encoded

    stft_encoded = layers.Lambda(lambda x: stft_encoding(x), output_shape=(None, fft_frame_size // 2 + 1))(inputs)

    # Magnitude and Phase extraction
    def extract_mag_phase(x):
        magnitude = tf.abs(x)
        phase = tf.math.angle(x)
        return tf.concat([tf.expand_dims(magnitude, axis=-1), tf.expand_dims(phase, axis=-1)], axis=-1)

    stft_mag_phase = layers.Lambda(lambda x: extract_mag_phase(x), output_shape=(None, fft_frame_size // 2 + 1, 2))(stft_encoded)

    # **Processing Path**
    x = layers.Conv2D(residual_channels, kernel_size=(3, 3), activation='relu', padding='same')(stft_mag_phase)
    skip_connections = []

    for _ in range(3):
        residual = x
        x = layers.Conv2D(residual_channels, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Add()([residual, x])
        skip = layers.Conv2D(skip_channels, kernel_size=(1, 1), activation='relu', padding='same')(x)
        skip_connections.append(skip)

    x = layers.Add()(skip_connections)
    x = layers.Conv2D(skip_channels, kernel_size=(1, 1), activation='relu', padding='same')(x)

    # Reconstructing magnitude and phase
    def reconstruct_mag_phase(x):
        magnitude, phase = tf.split(x, num_or_size_splits=2, axis=-1)
        return tf.complex(real=magnitude * tf.cos(phase), imag=magnitude * tf.sin(phase))

    stft_reconstructed = layers.Lambda(lambda x: reconstruct_mag_phase(x), output_shape=(None, fft_frame_size // 2 + 1))(x)

    # **ISTFT Decoding**
    def istft_decoding(x):
        # Split the input into real and imaginary parts for left and right channels
        real_left, real_right = tf.split(tf.math.real(x), num_or_size_splits=2, axis=-1)
        imag_left, imag_right = tf.split(tf.math.imag(x), num_or_size_splits=2, axis=-1)
        
        # Reconstruct complex tensors for both channels
        complex_left = tf.complex(real_left, imag_left)
        complex_right = tf.complex(real_right, imag_right)
        
        # Perform inverse STFT for both channels
        istft_left = tf.signal.inverse_stft(
            complex_left, 
            frame_length=fft_frame_size, 
            frame_step=fft_frame_size - fft_overlap,
            window_fn=tf.signal.hann_window
        )
        istft_right = tf.signal.inverse_stft(
            complex_right, 
            frame_length=fft_frame_size, 
            frame_step=fft_frame_size - fft_overlap,
            window_fn=tf.signal.hann_window
        )
        
        # Combine both channels
        stereo_output = tf.stack([istft_left, istft_right], axis=-1)        

        # Aggregate segments along the second dimension (reduce `8` to `1`)
        stereo_output = tf.reduce_mean(stereo_output, axis=1)        
        
        # Truncate to the target size (2048 samples per channel)
        stereo_output = stereo_output[:, :input_shape[0], :]        
        
        return stereo_output

    outputs = layers.Lambda(lambda x: istft_decoding(x), output_shape=input_shape)(stft_reconstructed)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="FFT_Audio_Model")

    return model


def build_fft_audio_model2(input_shape=(2048, 2), fft_frame_size=512, fft_overlap=256, dilations=[1, 2, 4, 8], residual_channels=64, skip_channels=128):
    inputs = layers.Input(shape=input_shape)

    # **STFT Encoding**
    def stft_encoding(x):
        stft_left = tf.signal.stft(x[:, :, 0], frame_length=fft_frame_size, frame_step=fft_frame_size - fft_overlap, pad_end=True)
        stft_right = tf.signal.stft(x[:, :, 1], frame_length=fft_frame_size, frame_step=fft_frame_size - fft_overlap, pad_end=True)
        stft_encoded = tf.complex(real=tf.math.real(stft_left), imag=tf.math.imag(stft_right))
        return stft_encoded

    stft_encoded = layers.Lambda(lambda x: stft_encoding(x), output_shape=(None, fft_frame_size // 2 + 1))(inputs)

    # Magnitude and Phase Extraction
    def extract_mag_phase(x):
        magnitude = tf.abs(x)
        phase = tf.math.angle(x)
        return tf.concat([tf.expand_dims(magnitude, axis=-1), tf.expand_dims(phase, axis=-1)], axis=-1)

    stft_mag_phase = layers.Lambda(lambda x: extract_mag_phase(x), output_shape=(None, fft_frame_size // 2 + 1, 2))(stft_encoded)

    # **Frequency-Domain Processing Pathway**
    freq_x = layers.Conv2D(residual_channels, kernel_size=(3, 3), activation='relu', padding='same')(stft_mag_phase)
    freq_skip_connections = []

    for _ in range(3):
        residual = freq_x
        freq_x = layers.Conv2D(residual_channels, kernel_size=(3, 3), activation='relu', padding='same')(freq_x)
        freq_x = layers.Add()([residual, freq_x])
        skip = layers.Conv2D(skip_channels, kernel_size=(1, 1), activation='relu', padding='same')(freq_x)
        freq_skip_connections.append(skip)

    freq_x = layers.Add()(freq_skip_connections)
    freq_x = layers.Conv2D(skip_channels, kernel_size=(1, 1), activation='relu', padding='same')(freq_x)

    # **Time-Domain Processing Pathway**
    time_x = layers.Conv1D(residual_channels, kernel_size=3, activation='relu', padding='same')(inputs)
    time_skip_connections = []

    for _ in range(3):
        residual = time_x
        time_x = layers.Conv1D(residual_channels, kernel_size=3, activation='relu', padding='same')(time_x)
        time_x = layers.Add()([residual, time_x])
        skip = layers.Conv1D(skip_channels, kernel_size=1, activation='relu', padding='same')(time_x)
        time_skip_connections.append(skip)

    time_x = layers.Add()(time_skip_connections)
    time_x = layers.Conv1D(skip_channels, kernel_size=1, activation='relu', padding='same')(time_x)
    # time_x shape: (None, 2048, 128)

    # Align time-domain features to match frequency-domain features
    time_x_reduced = layers.AveragePooling1D(pool_size=fft_frame_size - fft_overlap, strides=fft_frame_size - fft_overlap, padding='valid')(time_x)
    time_x_expanded = layers.Lambda(lambda x: tf.expand_dims(x, axis=2))(time_x_reduced)  # Expand time domain to add frequency dimension
    time_x_tiled = layers.Lambda(lambda x: tf.tile(x, [1, 1, fft_overlap + 1, 1]))(time_x_expanded)  # Tile to match frequency bins

    # Concatenate time and frequency features
    fused_x = layers.Concatenate()([freq_x, time_x_tiled])
    fused_x = layers.Conv2D(skip_channels, kernel_size=(3, 3), activation='relu', padding='same')(fused_x)    

    # Reconstructing Magnitude and Phase
    def reconstruct_mag_phase(x):
        magnitude, phase = tf.split(x, num_or_size_splits=2, axis=-1)
        return tf.complex(real=magnitude * tf.cos(phase), imag=magnitude * tf.sin(phase))

    stft_reconstructed = layers.Lambda(lambda x: reconstruct_mag_phase(x), output_shape=(None, fft_frame_size // 2 + 1))(fused_x)

    # **ISTFT Decoding**
    def istft_decoding(x):
        # Split the input into real and imaginary parts for left and right channels
        real_left, real_right = tf.split(tf.math.real(x), num_or_size_splits=2, axis=-1)
        imag_left, imag_right = tf.split(tf.math.imag(x), num_or_size_splits=2, axis=-1)
        
        # Reconstruct complex tensors for both channels
        complex_left = tf.complex(real_left, imag_left)
        complex_right = tf.complex(real_right, imag_right)
        
        # Perform inverse STFT for both channels
        istft_left = tf.signal.inverse_stft(
            complex_left, 
            frame_length=fft_frame_size, 
            frame_step=fft_frame_size - fft_overlap,
            window_fn=tf.signal.hann_window
        )
        istft_right = tf.signal.inverse_stft(
            complex_right, 
            frame_length=fft_frame_size, 
            frame_step=fft_frame_size - fft_overlap,
            window_fn=tf.signal.hann_window
        )
        
        # Combine both channels
        stereo_output = tf.stack([istft_left, istft_right], axis=-1)        

        # Aggregate segments along the second dimension (reduce `8` to `1`)
        stereo_output = tf.reduce_mean(stereo_output, axis=1)        
        
        # Truncate to the target size (2048 samples per channel)
        stereo_output = stereo_output[:, :input_shape[0], :]        
        
        return stereo_output

    outputs = layers.Lambda(lambda x: istft_decoding(x), output_shape=input_shape)(stft_reconstructed)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="FFT_Audio_Model_With_Fusion")

    return model


def time_domain_pathway(inputs: layers.Input, dilations=[1, 2, 4, 8], residual_channels=64, skip_channels=128):
    time_features = layers.Conv1D(residual_channels, kernel_size=1, activation=None, padding='same')(inputs)  # Initial projection
    skip_connections = []

    for dilation_rate in dilations:
        residual = time_features
        tanh_out = layers.Conv1D(residual_channels, kernel_size=3, dilation_rate=dilation_rate, padding='causal', activation='tanh')(time_features)
        sigmoid_out = layers.Conv1D(residual_channels, kernel_size=3, dilation_rate=dilation_rate, padding='causal', activation='sigmoid')(time_features)
        gating_signal = layers.Multiply()([tanh_out, sigmoid_out])

        residual_out = layers.Conv1D(residual_channels, kernel_size=1, activation=None, padding='same')(gating_signal)
        time_features = layers.Add()([residual, residual_out])

        skip_out = layers.Conv1D(skip_channels, kernel_size=1, activation=None, padding='same')(gating_signal)
        skip_connections.append(skip_out)

    time_skip_sum = layers.Add()(skip_connections)
    time_skip_sum = layers.Activation('relu')(time_skip_sum)
    time_skip_sum = layers.Conv1D(skip_channels, kernel_size=1, activation='relu', padding='same')(time_skip_sum)
    return time_skip_sum


def build_fft_audio_model3(input_shape=(2048, 2), fft_frame_size=512, fft_overlap=256, dilations=[1, 2, 4, 8], residual_channels=64, skip_channels=128):
    inputs = layers.Input(shape=input_shape)

    # **STFT Encoding**
    def stft_encoding(x):
        stft_left = tf.signal.stft(x[:, :, 0], frame_length=fft_frame_size, frame_step=fft_frame_size - fft_overlap, pad_end=True)
        stft_right = tf.signal.stft(x[:, :, 1], frame_length=fft_frame_size, frame_step=fft_frame_size - fft_overlap, pad_end=True)
        stft_encoded = tf.complex(real=tf.math.real(stft_left), imag=tf.math.imag(stft_right))
        return stft_encoded

    stft_encoded = layers.Lambda(lambda x: stft_encoding(x), output_shape=(None, fft_frame_size // 2 + 1))(inputs)

    # Magnitude and Phase Extraction
    def extract_mag_phase(x):
        magnitude = tf.abs(x)
        phase = tf.math.angle(x)
        return tf.concat([tf.expand_dims(magnitude, axis=-1), tf.expand_dims(phase, axis=-1)], axis=-1)

    stft_mag_phase = layers.Lambda(lambda x: extract_mag_phase(x), output_shape=(None, fft_frame_size // 2 + 1, 2))(stft_encoded)

    # **Frequency-Domain Processing Pathway**
    freq_x = layers.Conv2D(residual_channels, kernel_size=(3, 3), activation='relu', padding='same')(stft_mag_phase)
    freq_skip_connections = []    

    for dilation_rate in dilations:
        residual = freq_x
        tanh_out = layers.Conv2D(residual_channels, kernel_size=(3, 3), dilation_rate=dilation_rate, padding='same', activation='tanh')(freq_x)
        sigmoid_out = layers.Conv2D(residual_channels, kernel_size=(3, 3), dilation_rate=dilation_rate, padding='same', activation='sigmoid')(freq_x)                
        gating_signal = layers.Multiply()([tanh_out, sigmoid_out])
        residual_out = layers.Conv2D(residual_channels, kernel_size=(3, 3), activation='relu', padding='same')(freq_x)
        freq_x = layers.Add()([residual, residual_out])        
        skip_out = layers.Conv2D(skip_channels, kernel_size=(1, 1), activation=None, padding='same')(gating_signal)
        freq_skip_connections.append(skip_out)

    freq_x = layers.Add()(freq_skip_connections)
    freq_x = layers.Conv2D(skip_channels, kernel_size=(1, 1), activation='relu', padding='same')(freq_x)

    # **Time-Domain Processing Pathway**
    time_features = layers.Conv1D(residual_channels, kernel_size=1, activation=None, padding='same')(inputs)  # Initial projection
    skip_connections = []

    for dilation_rate in dilations:
        residual = time_features
        tanh_out = layers.Conv1D(residual_channels, kernel_size=3, dilation_rate=dilation_rate, padding='causal', activation='tanh')(time_features)
        sigmoid_out = layers.Conv1D(residual_channels, kernel_size=3, dilation_rate=dilation_rate, padding='causal', activation='sigmoid')(time_features)
        gating_signal = layers.Multiply()([tanh_out, sigmoid_out])

        residual_out = layers.Conv1D(residual_channels, kernel_size=1, activation=None, padding='same')(gating_signal)
        time_features = layers.Add()([residual, residual_out])

        skip_out = layers.Conv1D(skip_channels, kernel_size=1, activation=None, padding='same')(gating_signal)
        skip_connections.append(skip_out)

    time_x = layers.Add()(skip_connections)
    time_x = layers.Activation('relu')(time_x)
    time_x = layers.Conv1D(skip_channels, kernel_size=1, activation='relu', padding='same')(time_x)
    # time_x shape: (None, 2048, 128)

    # Align time-domain features to match frequency-domain features    
    time_x_aligned = layers.Conv1D(
        filters=128,  # Match the number of channels in the frequency domain
        kernel_size=fft_frame_size - fft_overlap,
        strides=fft_frame_size - fft_overlap,
        padding='valid',
        activation='relu'
    )(time_x)

    time_x_expanded = layers.Lambda(lambda x: tf.expand_dims(x, axis=2))(time_x_aligned)  # Expand time domain to add frequency dimension
    time_x_tiled = layers.Lambda(lambda x: tf.tile(x, [1, 1, fft_overlap + 1, 1]))(time_x_expanded)  # Tile to match frequency bins

    # Concatenate time and frequency features
    fused_x = layers.Concatenate()([freq_x, time_x_tiled])    
    fused_x = layers.Conv2D(skip_channels, kernel_size=(3, 3), activation='relu', padding='same')(fused_x)    

    # Reconstructing Magnitude and Phase
    def reconstruct_mag_phase(x):
        magnitude, phase = tf.split(x, num_or_size_splits=2, axis=-1)
        return tf.complex(real=magnitude * tf.cos(phase), imag=magnitude * tf.sin(phase))

    stft_reconstructed = layers.Lambda(lambda x: reconstruct_mag_phase(x), output_shape=(None, fft_frame_size // 2 + 1))(fused_x)

    # **ISTFT Decoding**
    def istft_decoding(x):
        # Split the input into real and imaginary parts for left and right channels
        real_left, real_right = tf.split(tf.math.real(x), num_or_size_splits=2, axis=-1)
        imag_left, imag_right = tf.split(tf.math.imag(x), num_or_size_splits=2, axis=-1)
        
        # Reconstruct complex tensors for both channels
        complex_left = tf.complex(real_left, imag_left)
        complex_right = tf.complex(real_right, imag_right)
        
        # Perform inverse STFT for both channels
        istft_left = tf.signal.inverse_stft(
            complex_left, 
            frame_length=fft_frame_size, 
            frame_step=fft_frame_size - fft_overlap,
            window_fn=tf.signal.hann_window
        )
        istft_right = tf.signal.inverse_stft(
            complex_right, 
            frame_length=fft_frame_size, 
            frame_step=fft_frame_size - fft_overlap,
            window_fn=tf.signal.hann_window
        )
        
        # Combine both channels
        stereo_output = tf.stack([istft_left, istft_right], axis=-1)        

        # Aggregate segments along the second dimension (reduce `8` to `1`)
        stereo_output = tf.reduce_mean(stereo_output, axis=1)        
        
        # Truncate to the target size (2048 samples per channel)
        stereo_output = stereo_output[:, :input_shape[0], :]        
        
        return stereo_output

    outputs = layers.Lambda(lambda x: istft_decoding(x), output_shape=input_shape)(stft_reconstructed)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="FFT_Audio_Model_With_Fusion")

    return model


def build_fft_audio_model_refined(input_shape=(2048, 2), fft_frame_size=512, fft_overlap=256,
                                   dilations=[1, 2, 4, 8], residual_channels=64, skip_channels=128):
    inputs = layers.Input(shape=input_shape)

    # **STFT Encoding**
    def stft_encoding(x):
        stft_left = tf.signal.stft(x[:, :, 0], frame_length=fft_frame_size, frame_step=fft_frame_size - fft_overlap, pad_end=True)
        stft_right = tf.signal.stft(x[:, :, 1], frame_length=fft_frame_size, frame_step=fft_frame_size - fft_overlap, pad_end=True)
        stft_encoded = tf.complex(real=tf.math.real(stft_left), imag=tf.math.imag(stft_right))
        return stft_encoded

    stft_encoded = layers.Lambda(lambda x: stft_encoding(x), output_shape=(None, fft_frame_size // 2 + 1))(inputs)

    # Magnitude and Phase Extraction
    def extract_mag_phase(x):
        magnitude = tf.abs(x)
        phase = tf.math.angle(x)
        return tf.concat([tf.expand_dims(magnitude, axis=-1), tf.expand_dims(phase, axis=-1)], axis=-1)

    stft_mag_phase = layers.Lambda(lambda x: extract_mag_phase(x), output_shape=(None, fft_frame_size // 2 + 1, 2))(stft_encoded)

    # **Frequency-Domain Processing Pathway**
    freq_x = layers.Conv2D(residual_channels, kernel_size=(3, 3), activation='relu', padding='same')(stft_mag_phase)
    freq_x = layers.Conv2D(residual_channels, kernel_size=(3, 3), activation='relu', padding='same')(freq_x)
    freq_skip_connections = []

    for dilation_rate in dilations:
        residual = freq_x
        tanh_out = layers.Conv2D(residual_channels, kernel_size=(3, 3), dilation_rate=dilation_rate, padding='same', activation='tanh')(freq_x)
        sigmoid_out = layers.Conv2D(residual_channels, kernel_size=(3, 3), dilation_rate=dilation_rate, padding='same', activation='sigmoid')(freq_x)
        gating_signal = layers.Multiply()([tanh_out, sigmoid_out])
        residual_out = layers.Conv2D(residual_channels, kernel_size=(3, 3), activation='relu', padding='same')(gating_signal)
        freq_x = layers.Add()([residual, residual_out])
        skip_out = layers.Conv2D(skip_channels, kernel_size=(1, 1), activation=None, padding='same')(gating_signal)
        freq_skip_connections.append(skip_out)

    freq_x = layers.Add()(freq_skip_connections)
    freq_x = layers.Conv2D(skip_channels, kernel_size=(1, 1), activation='relu', padding='same')(freq_x)

    # **Time-Domain Processing Pathway**
    time_features = layers.Conv1D(residual_channels, kernel_size=1, activation=None, padding='same')(inputs)
    skip_connections = []

    for dilation_rate in dilations:
        residual = time_features
        tanh_out = layers.Conv1D(residual_channels, kernel_size=3, dilation_rate=dilation_rate, padding='causal', activation='tanh')(time_features)
        sigmoid_out = layers.Conv1D(residual_channels, kernel_size=3, dilation_rate=dilation_rate, padding='causal', activation='sigmoid')(time_features)
        gating_signal = layers.Multiply()([tanh_out, sigmoid_out])
        residual_out = layers.Conv1D(residual_channels, kernel_size=1, activation=None, padding='same')(gating_signal)
        time_features = layers.Add()([residual, residual_out])
        skip_out = layers.Conv1D(skip_channels, kernel_size=1, activation=None, padding='same')(gating_signal)
        skip_connections.append(skip_out)

    time_x = layers.Add()(skip_connections)
    time_x = layers.Activation('relu')(time_x)
    time_x = layers.Conv1D(skip_channels, kernel_size=1, activation='relu', padding='same')(time_x)

    # Self-Attention for Time Features
    time_attention = layers.MultiHeadAttention(num_heads=4, key_dim=skip_channels)(query=time_x, value=time_x, key=time_x)
    time_x_aligned = layers.Conv1D(filters=skip_channels, kernel_size=fft_frame_size - fft_overlap, strides=fft_frame_size - fft_overlap, padding='valid', activation='relu')(time_attention)
    time_x_expanded = layers.Lambda(lambda x: tf.expand_dims(x, axis=2))(time_x_aligned)

    # **Fusion of Time and Frequency Features**
    freq_x_projected = layers.Conv2D(skip_channels, kernel_size=(1, 1), activation='relu', padding='same')(freq_x)
    time_x_projected = layers.Conv2D(skip_channels, kernel_size=(1, 1), activation='relu', padding='same')(time_x_expanded)
    fused_x = layers.Add()([freq_x_projected, time_x_projected])
    fused_x = layers.Conv2D(skip_channels, kernel_size=(3, 3), activation='relu', padding='same')(fused_x)

    # Reconstructing Magnitude and Phase
    def reconstruct_mag_phase(x):
        magnitude, phase = tf.split(x, num_or_size_splits=2, axis=-1)
        return tf.complex(real=magnitude * tf.cos(phase), imag=magnitude * tf.sin(phase))

    stft_reconstructed = layers.Lambda(lambda x: reconstruct_mag_phase(x), output_shape=(None, fft_frame_size // 2 + 1))(fused_x)

    # **ISTFT Decoding**
    def istft_decoding_with_alignment(x, fft_frame_size, fft_overlap, input_shape):
        """
        Decodes the ISTFT output while ensuring proper aggregation and truncation.
        """
        # Split real and imaginary parts for left and right channels
        real_left, real_right = tf.split(tf.math.real(x), num_or_size_splits=2, axis=-1)
        imag_left, imag_right = tf.split(tf.math.imag(x), num_or_size_splits=2, axis=-1)

        # Reconstruct complex tensors for both channels
        complex_left = tf.complex(real_left, imag_left)
        complex_right = tf.complex(real_right, imag_right)

        # Perform ISTFT for both channels
        istft_left = tf.signal.inverse_stft(
            complex_left, 
            frame_length=fft_frame_size, 
            frame_step=fft_frame_size - fft_overlap,
            window_fn=tf.signal.hann_window
        )
        istft_right = tf.signal.inverse_stft(
            complex_right, 
            frame_length=fft_frame_size, 
            frame_step=fft_frame_size - fft_overlap,
            window_fn=tf.signal.hann_window
        )

        # Combine left and right channels into stereo output
        stereo_output = tf.stack([istft_left, istft_right], axis=-1)

        # Aggregate overlapping segments along the window dimension (reduce `8` to `1`)
        stereo_output = tf.reduce_mean(stereo_output, axis=1)

        # Truncate to the target size (2048 samples per channel)
        stereo_output = stereo_output[:, :input_shape[0], :]

        return stereo_output

    # Usage in the model
    outputs = layers.Lambda(
        lambda x: istft_decoding_with_alignment(x, fft_frame_size, fft_overlap, input_shape),
        output_shape=input_shape
    )(stft_reconstructed)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="FFT_Audio_Model_Aligned")

    return model


def build_fft_audio_model5(input_shape=(2048, 2), fft_frame_size=512, fft_overlap=256, 
                          dilations=[1, 2, 4, 8], residual_channels=64, skip_channels=128, num_heads=4):
    inputs = layers.Input(shape=input_shape)

    # **STFT Encoding**
    def stft_encoding(x):
        stft_left = tf.signal.stft(x[:, :, 0], frame_length=fft_frame_size, frame_step=fft_frame_size - fft_overlap, pad_end=True)
        stft_right = tf.signal.stft(x[:, :, 1], frame_length=fft_frame_size, frame_step=fft_frame_size - fft_overlap, pad_end=True)
        stft_encoded = tf.complex(real=tf.math.real(stft_left), imag=tf.math.imag(stft_right))
        return stft_encoded

    stft_encoded = layers.Lambda(lambda x: stft_encoding(x))(inputs)

    # **Magnitude and Phase Extraction**
    def extract_mag_phase(x):
        magnitude = tf.math.log1p(tf.abs(x))  # Log-magnitude
        phase = tf.math.angle(x)
        phase_cos = tf.math.cos(phase)
        phase_sin = tf.math.sin(phase)
        return tf.concat([tf.expand_dims(magnitude, axis=-1), tf.expand_dims(phase_cos, axis=-1), tf.expand_dims(phase_sin, axis=-1)], axis=-1)

    stft_mag_phase = layers.Lambda(lambda x: extract_mag_phase(x))(stft_encoded)

    # **Frequency-Domain Processing Pathway**
    freq_x = layers.Conv2D(residual_channels, kernel_size=(3, 3), activation='relu', padding='same')(stft_mag_phase)
    freq_skip_connections = []

    for dilation_rate in dilations:
        residual = freq_x
        tanh_out = layers.Conv2D(residual_channels, kernel_size=(3, 3), dilation_rate=dilation_rate, padding='same', activation='tanh')(freq_x)
        sigmoid_out = layers.Conv2D(residual_channels, kernel_size=(3, 3), dilation_rate=dilation_rate, padding='same', activation='sigmoid')(freq_x)
        gating_signal = layers.Add()([
            layers.Multiply()([tanh_out, sigmoid_out]),
            layers.Conv2D(residual_channels, kernel_size=(1, 1), activation='relu', padding='same')(freq_x)
        ])
        residual_out = layers.Conv2D(residual_channels, kernel_size=(3, 3), activation='relu', padding='same')(freq_x)
        freq_x = layers.Add()([residual, residual_out])
        skip_out = layers.Conv2D(skip_channels, kernel_size=(1, 1), activation=None, padding='same')(gating_signal)
        freq_skip_connections.append(skip_out)

    freq_x = layers.Add()(freq_skip_connections)
    freq_x = layers.Conv2D(skip_channels, kernel_size=(1, 1), activation='relu', padding='same')(freq_x)

    # **Time-Domain Processing Pathway**
    time_features = layers.Conv1D(residual_channels, kernel_size=1, activation=None, padding='same')(inputs)  # Initial projection
    skip_connections = []

    for dilation_rate in dilations:
        residual = time_features
        tanh_out = layers.Conv1D(residual_channels, kernel_size=3, dilation_rate=dilation_rate, padding='causal', activation='tanh')(time_features)
        sigmoid_out = layers.Conv1D(residual_channels, kernel_size=3, dilation_rate=dilation_rate, padding='causal', activation='sigmoid')(time_features)
        gating_signal = layers.Multiply()([tanh_out, sigmoid_out])
        residual_out = layers.Conv1D(residual_channels, kernel_size=1, activation=None, padding='same')(gating_signal)
        time_features = layers.Add()([residual, residual_out])
        skip_out = layers.Conv1D(skip_channels, kernel_size=1, activation=None, padding='same')(gating_signal)
        skip_connections.append(skip_out)

    time_x = layers.Add()(skip_connections)
    time_x = layers.Activation('relu')(time_x)
    time_x = layers.Conv1D(skip_channels, kernel_size=1, activation='relu', padding='same')(time_x)

    # Align time-domain features to match frequency-domain features
    time_x_aligned = layers.Conv1D(
        filters=skip_channels,  # Match the number of channels in the frequency domain
        kernel_size=fft_frame_size - fft_overlap,
        strides=fft_frame_size - fft_overlap,
        padding='valid',
        activation='relu'
    )(time_x)

    time_x_expanded = layers.Lambda(lambda x: tf.expand_dims(x, axis=2))(time_x_aligned)  # Expand time domain to add frequency dimension

    # **Attention-Based Fusion**
    def attention_fusion(time_features, freq_features):
        time_shape = tf.shape(time_features)
        freq_shape = tf.shape(freq_features)
        time_flat = tf.reshape(time_features, [time_shape[0], -1, time_shape[-1]])
        freq_flat = tf.reshape(freq_features, [freq_shape[0], -1, freq_shape[-1]])
        combined_features = tf.concat([time_flat, freq_flat], axis=1)
        attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=time_flat.shape[-1])(combined_features, combined_features)
        time_attention = tf.reshape(attention[:, :time_shape[1], :], time_shape)
        freq_attention = tf.reshape(attention[:, time_shape[1]:, :], freq_shape)
        return time_attention, freq_attention

    time_x_fused, freq_x_fused = layers.Lambda(lambda x: attention_fusion(x[0], x[1]))([time_x_expanded, freq_x])

    # Concatenate fused features
    # Ensure time_x_fused matches the shape of freq_x_fused along the frequency dimension
    time_x_fused_expanded = layers.Lambda(
        lambda inputs: tf.tile(inputs[0], [1, 1, tf.shape(inputs[1])[2], 1]),
        output_shape=lambda input_shape: (input_shape[0][0], input_shape[0][1], freq_x_fused.shape[2], input_shape[0][-1])
    )([time_x_fused, freq_x_fused])

    # Concatenate the aligned features
    fused_x = layers.Concatenate()([freq_x_fused, time_x_fused_expanded])
    
    # Reduce the last dimension to 3 using Conv2D
    fused_x = layers.Conv2D(2, kernel_size=(1, 1), activation=None, padding='same')(fused_x)

    # **Reconstruct Magnitude and Phase**
    def reconstruct_mag_phase(x):
        # x has shape (batch_size, num_frames, fft_bins, 2) after Conv2D with 2 filters
        magnitude, phase = tf.split(x, num_or_size_splits=2, axis=-1)
        magnitude = tf.squeeze(magnitude, axis=-1)  # Remove last dimension
        phase = tf.squeeze(phase, axis=-1)  # Remove last dimension
        return tf.complex(real=magnitude * tf.cos(phase), imag=magnitude * tf.sin(phase))

    stft_reconstructed = layers.Lambda(
        lambda x: reconstruct_mag_phase(x),
        output_shape=lambda input_shape: input_shape[:-1] + (2,)  # Last dimension becomes 2 (real, imag)
    )(fused_x)

    #print(f"Shape of stft_reconstructed: {stft_reconstructed.shape}")

    # **ISTFT Decoding**
    def istft_decoding(x):
        # Apply ISTFT (produces a real-valued signal)
        time_signal = tf.signal.inverse_stft(
            x,
            frame_length=fft_frame_size,
            frame_step=fft_frame_size - fft_overlap,
            window_fn=tf.signal.hann_window
        )
        # Truncate to match the original input length
        time_signal = time_signal[:, :input_shape[0]]

        # Duplicate the signal along the last axis to produce two channels
        return tf.stack([time_signal, time_signal], axis=-1)

    outputs = layers.Lambda(
        lambda x: istft_decoding(x),
        output_shape=input_shape  # Single channel (real-valued output)
    )(stft_reconstructed)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="FFT_Audio_Model_With_Fusion")

    return model
