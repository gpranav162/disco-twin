import tensorflow as tf
from tensorflow.keras import layers, Model

def build_echelon_classifier(input_shape: tuple, num_classes: int) -> Model:
    inputs = layers.Input(shape=input_shape)
    
    x = layers.LSTM(16, return_sequences=True, dropout=0.1)(inputs)
    x = layers.LSTM(16, return_sequences=False, dropout=0.1)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model