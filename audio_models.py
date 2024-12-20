import os
import numpy as np
import tensorflow as tf
# VS-Code Python debugger can't resolve this, but should still work
#from tensorflow.keras import layers
# instead you could 'pip install keras' and use
from keras import layers

import audio_config as cfg

#from tensorflow.python.platform.tf_logging import fatal

def build_model():
    ''' 
    global function to build the model for training and inference.
    Modify here to load another model.
    '''
    input_shape = (cfg.InputSize, cfg.NumChannels)    
    model = build_wavelet_wavenet(input_shape=input_shape, dilations=[1,2,4,8,16], residual_channels=128, skip_channels=256)
    model = compile_model(model)
    model.summary()
    return model

def my_loss(y_true, y_pred):
    loss = combined_loss(y_true, y_pred, alpha=1.0, beta=0.005, gamma=0, frame_length=2048, frame_step=512)    
    return loss

def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.LearningRate)    
    model.compile(optimizer=optimizer, loss=my_loss, metrics=['accuracy', 'mse'])
    return model

def load_weights(model, bitrate: str = None):
    model_file_name = cfg.ModelFileName(bitrate)
    if not os.path.exists(model_file_name):
        # fall back to default model name
        model_file_name = cfg.ModelFileName()
    if os.path.exists(model_file_name):
        print(f"Loading model weights from '{model_file_name}'")
        model.load_weights(model_file_name)
    else:
        print(f"Model {model_file_name} not found.")
  

def build_wavenet(input_shape=(2048, 2), dilations=[1, 2, 4, 8]) -> tf.keras.Model:    
    inputs = layers.Input(shape=input_shape)
    x = inputs
    
    # Causal Convolutions
    for d in dilations:
        residual = x
        out_conv = layers.Conv1D(64, 3, padding='causal', activation='relu')(x)
        out_gate = layers.Conv1D(64, 3, padding='causal', activation='sigmoid')(out_conv)
        gating_signal = layers.Multiply()([out_conv, out_gate])
        
        # Transform residual to match gating_signal channels
        residual_transformed = layers.Conv1D(64, 1, padding='same')(residual)
        residual = layers.Add()([residual_transformed, gating_signal])  # Skip connection
        
        x = layers.Conv1D(64, 1)(residual)  # Dilated Residual Network
    
    output = layers.Conv1D(input_shape[1], kernel_size=3, padding='causal', activation='tanh')(x)

    # Output Shape Adjustment
    final_output = layers.Reshape(input_shape)(output)
    
    model = tf.keras.Model(inputs=inputs, outputs=final_output, name="AudioWaveNet")
    
    return model


def build_advanced_wavenet(input_shape=(2048, 2), dilations=[1, 2, 4, 8], residual_channels=64, skip_channels=128) -> tf.keras.Model:
    """
    Advanced WaveNet implementation with residual and skip connections.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(residual_channels, kernel_size=1, activation=None, padding='same')(inputs)  # Initial projection
    skip_connections = []

    for dilation_rate in dilations:
        # Residual Block
        residual = x
        tanh_out = layers.Conv1D(residual_channels, kernel_size=3, dilation_rate=dilation_rate, padding='causal', activation='tanh')(x)
        sigmoid_out = layers.Conv1D(residual_channels, kernel_size=3, dilation_rate=dilation_rate, padding='causal', activation='sigmoid')(x)
        gating_signal = layers.Multiply()([tanh_out, sigmoid_out])  # Gated activation unit

        # Residual connection
        residual_out = layers.Conv1D(residual_channels, kernel_size=1, activation=None, padding='same')(gating_signal)
        x = layers.Add()([residual, residual_out])

        # Skip connection
        skip_out = layers.Conv1D(skip_channels, kernel_size=1, activation=None, padding='same')(gating_signal)
        skip_connections.append(skip_out)

    # Combine all skip connections
    skip_sum = layers.Add()(skip_connections)
    skip_sum = layers.Activation('relu')(skip_sum)
    skip_sum = layers.Conv1D(skip_channels, kernel_size=1, activation='relu', padding='same')(skip_sum)
    skip_sum = layers.Conv1D(input_shape[1], kernel_size=1, activation='tanh', padding='same')(skip_sum)

    # Output shape adjustment
    final_output = layers.Reshape(input_shape)(skip_sum)
    model = tf.keras.Model(inputs=inputs, outputs=final_output, name="AdvancedWaveNet")
    
    return model

def wavelet_transform_conv1d(input_tensor):
    """
    Applies a wavelet-like transform using a Conv1D layer.
    """
    filters = 4  # Number of wavelet filters
    wavelet_conv = layers.Conv1D(filters, kernel_size=5, strides=2, padding="same", activation="relu")
    return wavelet_conv(input_tensor)

def build_complex_wavenet(input_shape=(2048, 2), dilations=[1, 2, 4, 8], residual_channels=64, skip_channels=128) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    
    # **Wavelet Transform Pathway**
    wavelet_features = wavelet_transform_conv1d(inputs)
    
    # **Time-Domain Pathway**
    time_features = layers.Conv1D(residual_channels, kernel_size=1, activation=None, padding='same')(inputs)  # Initial projection
    skip_connections = []

    for dilation_rate in dilations:
        # Residual Block
        residual = time_features
        tanh_out = layers.Conv1D(residual_channels, kernel_size=3, dilation_rate=dilation_rate, padding='causal', activation='tanh')(time_features)
        sigmoid_out = layers.Conv1D(residual_channels, kernel_size=3, dilation_rate=dilation_rate, padding='causal', activation='sigmoid')(time_features)
        gating_signal = layers.Multiply()([tanh_out, sigmoid_out])  # Gated activation unit

        # Residual connection
        residual_out = layers.Conv1D(residual_channels, kernel_size=1, activation=None, padding='same')(gating_signal)
        time_features = layers.Add()([residual, residual_out])

        # Skip connection
        skip_out = layers.Conv1D(skip_channels, kernel_size=1, activation=None, padding='same')(gating_signal)
        skip_connections.append(skip_out)

    # Combine skip connections in the time domain
    time_skip_sum = layers.Add()(skip_connections)
    time_skip_sum = layers.Activation('relu')(time_skip_sum)
    time_skip_sum = layers.Conv1D(skip_channels, kernel_size=1, activation='relu', padding='same')(time_skip_sum)

    # **Upsample Wavelet Features to Match Time Dimension**
    wavelet_features_upsampled = layers.UpSampling1D(size=2)(wavelet_features)

    # **Fusion of Time and Frequency Features**
    fusion_features = layers.Concatenate(axis=-1)([time_skip_sum, wavelet_features_upsampled])
    fusion_features = layers.Conv1D(skip_channels, kernel_size=1, activation='relu', padding='same')(fusion_features)
    
    # Final Projection
    output = layers.Conv1D(input_shape[1], kernel_size=1, activation='tanh', padding='same')(fusion_features)

    # Output shape adjustment
    final_output = layers.Reshape(input_shape)(output)
    model = tf.keras.Model(inputs=inputs, outputs=final_output, name="ComplexWaveNet")
    
    return model
    
#############################

def wavelet_transform_tf(input_tensor, wavelet='haar'):
    """
    Applies a discrete wavelet transform using TensorFlow.

    Parameters:
    - input_tensor: A TensorFlow tensor with shape (batch_size, time_steps, channels).
    - wavelet: The wavelet type, e.g., 'haar', 'db1', 'sym2', etc.

    Returns:
    - A transformed tensor with the wavelet coefficients.
    """
    # Define the filters for supported wavelets
    wavelet_filters = {
        'haar': ([1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]),
        'db1': ([1 / np.sqrt(2), 1 / np.sqrt(2)], [-1 / np.sqrt(2), 1 / np.sqrt(2)]),
        'sym2': ([0.48296, 0.83652, 0.22414, -0.12941], 
                 [-0.12941, -0.22414, 0.83652, -0.48296]),
    }
    
    if wavelet not in wavelet_filters:
        raise ValueError(f"Unsupported wavelet type: {wavelet}")
    
    # Get the decomposition filters for the selected wavelet
    dec_lo, dec_hi = wavelet_filters[wavelet]
    
    # Convert to TensorFlow constants
    dec_lo = tf.constant(dec_lo, dtype=tf.float32)
    dec_hi = tf.constant(dec_hi, dtype=tf.float32)
    
    # Match the filter shapes to the input channels
    input_channels = input_tensor.shape[-1]
    #kernel_size = len(dec_lo)
    kernel_size = dec_lo.shape[0]
    dec_lo = tf.reshape(dec_lo, (kernel_size, 1, 1))  # Shape: (kernel_size, 1, 1)
    dec_hi = tf.reshape(dec_hi, (kernel_size, 1, 1))
    
    # Broadcast filters to match the input channels
    dec_lo = tf.tile(dec_lo, [1, 1, input_channels])  # Shape: (kernel_size, 1, input_channels)
    dec_hi = tf.tile(dec_hi, [1, 1, input_channels])
    
    # Apply the filters using convolution
    approx = tf.nn.conv1d(input_tensor, filters=dec_lo, stride=2, padding="VALID")
    detail = tf.nn.conv1d(input_tensor, filters=dec_hi, stride=2, padding="VALID")
    
    # Concatenate the approximation and detail coefficients along the channel dimension
    return tf.concat([approx, detail], axis=-1)


def calculate_wavelet_output_shape(input_shape, wavelet='haar'):
    """
    Computes the output shape of the wavelet transform for a given input shape.
    
    Parameters:
    - input_shape: Tuple (time_steps, channels).
    - wavelet: Wavelet type, e.g., 'haar', 'db1', 'sym2'. Default is 'haar'.
    
    Returns:
    - Tuple of the output shape (time_steps, 2 * channels) to include approximation and detail coefficients.
    """
    time_steps, channels = input_shape
    
    # Define the wavelet filter lengths for supported wavelets
    wavelet_filters = {
        'haar': [1 / np.sqrt(2), 1 / np.sqrt(2)],  # Haar wavelet
        'db1': [1 / np.sqrt(2), 1 / np.sqrt(2)],  # Daubechies 1 wavelet
        'sym2': [0.48296, 0.83652, 0.22414, -0.12941],  # Symlet 2 wavelet
    }
    
    if wavelet not in wavelet_filters:
        raise ValueError(f"Unsupported wavelet type: {wavelet}")
    
    filter_length = len(wavelet_filters[wavelet])  # Length of the wavelet filter
    
    # Align with wavelet padding behavior: Add filter_length - 1 to input time steps
    padded_time_steps = time_steps + (filter_length - 1)
    
    # Calculate the reduced time steps after downsampling
    output_time_steps = padded_time_steps // 2
    
    # Output includes both approximation (cA) and detail (cD) coefficients
    return (output_time_steps, 2 * channels)


# Custom layer for dynamic padding
class DynamicPaddingLayer(layers.Layer):
    def __init__(self):
        super(DynamicPaddingLayer, self).__init__()

    def call(self, wavelet_features_resized, time_skip_sum):
        # Get dynamic time dimension from time_skip_sum
        time_skip_length = tf.shape(time_skip_sum)[1]  # Extract time dimension (axis=1)
        wavelet_features_length = tf.shape(wavelet_features_resized)[1]  # Get wavelet features time dimension

        # Calculate padding length dynamically
        pad_length = time_skip_length - wavelet_features_length

        # Apply padding with dynamic length along the time dimension
        padded_wavelet_features = tf.pad(wavelet_features_resized, [[0, 0], [0, pad_length], [0, 0]])
        return padded_wavelet_features

    def compute_output_shape(self, input_shape):
        # Handle dynamic input shapes
        if input_shape[0] is None or input_shape[1] is None:
            # If any input dimension is None (batch size or time steps), return dynamic shape
            return (input_shape[0], input_shape[1], input_shape[2])  # Same batch size and features, adjusted time steps

        # Otherwise, handle the shapes directly (non-dynamic dimensions)
        wavelet_shape, time_skip_shape = input_shape

        # Output shape should have the same batch size and feature dimensions, with adjusted time dimension
        return (wavelet_shape[0], time_skip_shape[1], wavelet_shape[2])

# Our winning model
def build_wavelet_wavenet(input_shape=(2048, 2), dilations=[1, 2, 4, 8], residual_channels=64, skip_channels=128) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)        

    # **Wavelet Transform Pathway**
    wavelet_output_shape = calculate_wavelet_output_shape(input_shape)  # Calculate output shape        
    wavelet_features = layers.Lambda(
        lambda x: wavelet_transform_tf(x, wavelet='haar'),
        #lambda x: wavelet_transform_tf(x, wavelet='sym2'),
        output_shape=(None, wavelet_output_shape[0], wavelet_output_shape[1])  # Batch dimension remains None
    )(inputs)

    # **Reshaping wavelet features**
    wavelet_features_reshaped = layers.Reshape((-1, wavelet_output_shape[1]))(wavelet_features)

    # **Time-Domain Pathway**
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
    # time_skip_sum shape: (None, 2048, 128)

    # **Wavelet Features Processing**
    wavelet_features_resized = layers.Conv1D(skip_channels, kernel_size=1, activation='relu', padding='same')(wavelet_features_reshaped)

    # **Dynamic Padding**
    wavelet_features_resized_padded = DynamicPaddingLayer()(wavelet_features_resized, time_skip_sum)

    # **Fusion of Time and Frequency Features**
    fusion_features = layers.Concatenate(axis=-1)([time_skip_sum, wavelet_features_resized_padded])

    # Ensure the dimensions of fusion_features are fixed before passing to Conv1D
    fusion_features = layers.Conv1D(skip_channels, kernel_size=1, activation='relu', padding='same')(fusion_features)

    # Final Projection
    output = layers.Conv1D(input_shape[1], kernel_size=1, activation='tanh', padding='same')(fusion_features)

    # Output shape adjustment
    final_output = layers.Reshape(input_shape)(output)
    model = tf.keras.Model(inputs=inputs, outputs=final_output, name="ComplexWaveNet")

    return model

############ ADVANCED LOSS FUNCTIONS ###############

def stft_loss(y_true, y_pred, frame_length=2048, frame_step=512):
    # Convert to mono for STFT
    y_true = tf.reduce_mean(y_true, axis=-1)  # Shape: [batch_size, signal_length]
    y_pred = tf.reduce_mean(y_pred, axis=-1)
    
    # Perform STFT
    y_true_stft = tf.signal.stft(y_true, frame_length=frame_length, frame_step=frame_step, pad_end=True)
    y_pred_stft = tf.signal.stft(y_pred, frame_length=frame_length, frame_step=frame_step, pad_end=True)

    # Compute magnitude spectrogram
    y_true_mag = tf.abs(y_true_stft)
    y_pred_mag = tf.abs(y_pred_stft)

    # Mean squared error in frequency domain
    loss = tf.reduce_mean(tf.square(y_true_mag - y_pred_mag))
    return loss

def high_freq_emphasis_loss(y_true, y_pred, frame_length=2048, frame_step=512):
    # Convert to mono for STFT
    y_true = tf.reduce_mean(y_true, axis=-1)  # Shape: [batch_size, signal_length]
    y_pred = tf.reduce_mean(y_pred, axis=-1)    

    # Compute STFT for true and predicted signals
    y_true_stft = tf.signal.stft(y_true, frame_length=frame_length, frame_step=frame_step, pad_end=True)
    y_pred_stft = tf.signal.stft(y_pred, frame_length=frame_length, frame_step=frame_step, pad_end=True)
    
    # Compute magnitude spectrograms
    y_true_mag = tf.abs(y_true_stft)
    y_pred_mag = tf.abs(y_pred_stft)
    
    # Frequency weighting (linear emphasis on high frequencies)
    freq_bins = tf.range(tf.shape(y_true_stft)[-1], dtype=tf.float32)
    freq_weights = freq_bins / tf.reduce_max(freq_bins)
    
    # Apply frequency weighting
    weighted_loss = freq_weights * tf.square(y_true_mag - y_pred_mag)

    result = tf.reduce_mean(weighted_loss)    
    return result

def combined_loss(y_true, y_pred, alpha=1.0, beta=0.005, gamma=0.005, frame_length=2048, frame_step=512):
    # Time-domain loss (MSE)
    time_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Frequency-domain losses
    stft_loss_value = 0.0
    if beta > 0.0:
        stft_loss_value = stft_loss(y_true, y_pred, frame_length=frame_length, frame_step=frame_step)    

    freq_loss_value = 0.0
    if gamma > 0.0:
        freq_loss_value = high_freq_emphasis_loss(y_true, y_pred, frame_length=frame_length, frame_step=frame_step)

    # Weighted combination
    return alpha * time_loss + beta * stft_loss_value + gamma * freq_loss_value    

'''
def combined_loss_2(y_true, y_pred):
    time_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    fft_true = tf.signal.stft(y_true[..., 0], frame_length=512, frame_step=256)
    fft_pred = tf.signal.stft(y_pred[..., 0], frame_length=512, frame_step=256)
    freq_loss = tf.reduce_mean(tf.square(tf.abs(fft_true) - tf.abs(fft_pred)))
    #return time_loss + 0.5 * freq_loss
    return time_loss + 0.005 * freq_loss
'''

def build_wavelet_wavenet_huge(
    input_shape=(2048, 2), dilations=[1, 2, 4, 8, 16, 32], 
    residual_channels=256, skip_channels=512
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)

    # **Wavelet Transform Pathway**
    wavelet_output_shape = calculate_wavelet_output_shape(input_shape)  # Calculate output shape        
    wavelet_features = layers.Lambda(
        lambda x: wavelet_transform_tf(x, wavelet='haar'),
        output_shape=(None, wavelet_output_shape[0], wavelet_output_shape[1])  # Batch dimension remains None
    )(inputs)

    # **Reshaping wavelet features**
    wavelet_features_reshaped = layers.Reshape((-1, wavelet_output_shape[1]))(wavelet_features)

    # **Time-Domain Pathway**
    time_features = layers.Conv1D(residual_channels, kernel_size=3, activation=None, padding='same')(inputs)  # Initial projection
    skip_connections = []

    for _ in range(2):  # Add two stacks of dilations for greater depth
        for dilation_rate in dilations:
            residual = time_features
            tanh_out = layers.Conv1D(residual_channels, kernel_size=5, dilation_rate=dilation_rate, padding='causal', activation='tanh')(time_features)
            sigmoid_out = layers.Conv1D(residual_channels, kernel_size=5, dilation_rate=dilation_rate, padding='causal', activation='sigmoid')(time_features)
            gating_signal = layers.Multiply()([tanh_out, sigmoid_out])

            residual_out = layers.Conv1D(residual_channels, kernel_size=1, activation=None, padding='same')(gating_signal)
            time_features = layers.Add()([residual, residual_out])

            skip_out = layers.Conv1D(skip_channels, kernel_size=1, activation=None, padding='same')(gating_signal)
            skip_connections.append(skip_out)

    time_skip_sum = layers.Add()(skip_connections)
    time_skip_sum = layers.Activation('relu')(time_skip_sum)
    time_skip_sum = layers.Conv1D(skip_channels, kernel_size=1, activation='relu', padding='same')(time_skip_sum)

    # **Wavelet Features Processing**
    wavelet_features_resized = layers.Conv1D(skip_channels, kernel_size=1, activation='relu', padding='same')(wavelet_features_reshaped)

    # **Dynamic Padding**
    wavelet_features_resized_padded = DynamicPaddingLayer()(wavelet_features_resized, time_skip_sum)

    # **Fusion of Time and Frequency Features**
    fusion_features = layers.Concatenate(axis=-1)([time_skip_sum, wavelet_features_resized_padded])

    # Additional processing for fusion
    fusion_features = layers.Conv1D(skip_channels, kernel_size=3, activation='relu', padding='same')(fusion_features)
    fusion_features = layers.Conv1D(skip_channels, kernel_size=3, activation='relu', padding='same')(fusion_features)

    # Final Projection
    output = layers.Conv1D(input_shape[1], kernel_size=1, activation='tanh', padding='same')(fusion_features)

    # Output shape adjustment
    final_output = layers.Reshape(input_shape)(output)
    model = tf.keras.Model(inputs=inputs, outputs=final_output, name="ComplexWaveNet_Large")

    return model