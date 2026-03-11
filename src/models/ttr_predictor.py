import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l1

def build_ttr_predictor(input_shape: tuple) -> Model:
    inputs = layers.Input(shape=input_shape)
    
    # L1 regularization with a factor of 10^-3 on the first layer
    x = layers.LSTM(64, return_sequences=True, dropout=0.1, 
                    kernel_regularizer=l1(1e-3))(inputs)
    x = layers.LSTM(64, return_sequences=False, dropout=0.1)(x)
    
    # Linear activation for continuous time prediction
    outputs = layers.Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])
    
    return model