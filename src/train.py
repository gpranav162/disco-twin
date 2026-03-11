import numpy as np
import pandas as pd
import joblib
import os
from data_preprocessing import preprocess_pipeline
from models.autoencoder import build_autoencoder, get_reconstruction_error
from models.ocsvm_detector import AnomalyDetector
from models.echelon_classifier import build_echelon_classifier
from models.ttr_predictor import build_ttr_predictor

def main():
    # Make sure save directory exists
    os.makedirs('../data/saved_models', exist_ok=True)

    # 1. Load Data
    print("Loading data...")
    # df = pd.DataFrame(np.random.rand(2000, 13))
    
    data_path = '../data/raw/supply_chain_data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run simulation.py first.")
        
    df = pd.read_csv(data_path)
    
    # Drop the 'time' column if it was saved, as the autoencoder only wants the raw features
    if 'time' in df.columns:
        df = df.drop(columns=['time'])
        
    X_train, X_val, X_test, scaler = preprocess_pipeline(df, window_size=14)
    
    # Save scaler for later evaluation
    joblib.dump(scaler, '../data/saved_models/scaler.pkl')

    # 2. Train Autoencoder
    print("Training Autoencoder...")
    autoencoder = build_autoencoder(input_dim=X_train.shape[1])
    autoencoder.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=1000, # [cite: 397]
        batch_size=128, # [cite: 396]
        verbose=0
    )
    autoencoder.save('../data/saved_models/autoencoder.h5')

    # 3. Train OCSVM
    print("Training OCSVM...")
    train_errors = get_reconstruction_error(autoencoder, X_train)
    detector = AnomalyDetector(nu=0.025, gamma=100.0) # [cite: 399]
    detector.fit(train_errors)
    joblib.dump(detector, '../data/saved_models/ocsvm_detector.pkl')

    # 4. Train Disrupted Echelon Classifier
    print("Training Echelon Classifier...")
    # Dummy labels for classifier (5 classes: Normal + 4 Disruption scenarios)
    y_class_train = tf.keras.utils.to_categorical(np.random.randint(0, 5, len(X_train)), num_classes=5)
    y_class_val = tf.keras.utils.to_categorical(np.random.randint(0, 5, len(X_val)), num_classes=5)
    
    # Reshape X for LSTM (samples, timesteps, features)
    X_train_lstm = X_train.reshape((X_train.shape[0], 14, 13))
    X_val_lstm = X_val.reshape((X_val.shape[0], 14, 13))

    classifier = build_echelon_classifier(input_shape=(14, 13), num_classes=5)
    classifier.fit(
        X_train_lstm, y_class_train,
        validation_data=(X_val_lstm, y_class_val),
        epochs=20, # [cite: 528]
        batch_size=32, # [cite: 528]
        verbose=0
    )
    classifier.save('../data/saved_models/echelon_classifier.h5')

    # 5. Train TTR Predictors (4 separate models)
    print("Training TTR Predictors...")
    for scenario in range(1, 5):
        # Dummy TTR target variable
        y_ttr_train = np.random.rand(len(X_train_lstm), 1) * 100 
        y_ttr_val = np.random.rand(len(X_val_lstm), 1) * 100

        ttr_model = build_ttr_predictor(input_shape=(14, 13))
        ttr_model.fit(
            X_train_lstm, y_ttr_train,
            validation_data=(X_val_lstm, y_ttr_val),
            epochs=20, # [cite: 611]
            batch_size=16, # [cite: 611]
            verbose=0
        )
        ttr_model.save(f'../data/saved_models/ttr_predictor_s{scenario}.h5')

    print("All models trained and saved successfully.")

if __name__ == "__main__":
    main()