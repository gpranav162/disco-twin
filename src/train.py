import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
# Ensure these local imports work by running from the correct directory
from data_preprocessing import preprocess_pipeline
from models.autoencoder import build_autoencoder, get_reconstruction_error
from models.ocsvm_detector import AnomalyDetector
from models.echelon_classifier import build_echelon_classifier
from models.ttr_predictor import build_ttr_predictor

def main():
    # --- 1. SET UP PATHS ---
    # Change 'My_Project_Folder' to your actual folder name in Drive
    BASE_PATH = '/content/disco-twin'
    DATA_PATH = '/content/drive/MyDrive/fyp digital supply chain/supply_chain_data.csv'
    SAVE_DIR = os.path.join(BASE_PATH, 'saved_models')

    # Create the directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        print(f"Creating directory: {SAVE_DIR}")
        os.makedirs(SAVE_DIR, exist_ok=True)

    # --- 2. LOAD DATA ---
    print(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing CSV! Path checked: {DATA_PATH}")
        
    df = pd.read_csv(DATA_PATH)
    if 'time' in df.columns:
        df = df.drop(columns=['time'])
        
    X_train, X_val, X_test, scaler = preprocess_pipeline(df, window_size=14)
    joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler.pkl'))

    # --- 3. TRAIN & SAVE ---
    # Autoencoder
    print("Training Autoencoder...")
    autoencoder = build_autoencoder(input_dim=X_train.shape[1])
    autoencoder.fit(X_train, X_train, validation_data=(X_val, X_val), epochs=50, verbose=1)
    autoencoder.save(os.path.join(SAVE_DIR, 'autoencoder.h5'))

    # OCSVM
    print("Training OCSVM...")
    train_errors = get_reconstruction_error(autoencoder, X_train)
    detector = AnomalyDetector(nu=0.025, gamma=100.0)
    detector.fit(train_errors)
    joblib.dump(detector, os.path.join(SAVE_DIR, 'ocsvm_detector.pkl'))

    # ... (Repeat os.path.join(SAVE_DIR, ...) for Echelon and TTR models)
    
    print(f"Success! Models saved in: {SAVE_DIR}")

if __name__ == "__main__":
    main()