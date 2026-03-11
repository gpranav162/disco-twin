import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

def build_autoencoder(input_dim: int = 182) -> Model:
    """
    Default input_dim is 182 
    (14 timesteps * 13 features).
    """
    encoder_input = layers.Input(shape=(input_dim,))
    
    x = layers.Dense(256, activation='relu')(encoder_input)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    
    latent = layers.Dense(32, activation='relu')(x)
    
    x = layers.Dense(64, activation='relu')(latent)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    
    decoder_output = layers.Dense(input_dim, activation='linear')(x)
    
    autoencoder = Model(inputs=encoder_input, outputs=decoder_output)
    
    # MAE loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    autoencoder.compile(optimizer=optimizer, loss='mae')
    
    return autoencoder

def get_reconstruction_error(model: Model, data: np.ndarray) -> np.ndarray:
    """
    Absolute reconstruction error.
    """
    reconstructions = model.predict(data, batch_size=128)
    absolute_error = np.abs(data - reconstructions)
    
    return absolute_error