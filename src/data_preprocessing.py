import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

def train_val_test_split(
    df: pd.DataFrame, 
    train_ratio: float = 0.6, 
    val_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

def scale_data(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    
    scaler = MinMaxScaler()
    
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)
    
    return train_scaled, val_scaled, test_scaled, scaler

def create_sliding_window(
    data: np.ndarray, 
    window_size: int = 14, 
    flatten: bool = True
) -> np.ndarray:
    
    sequences = []
    
    for i in range(len(data) - window_size + 1):
        window = data[i:(i + window_size)]
        if flatten:
            window = window.flatten()
        sequences.append(window)
        
    return np.array(sequences)

def preprocess_pipeline(
    df: pd.DataFrame, 
    window_size: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    
    train_df, val_df, test_df = train_val_test_split(df)
    
    train_scaled, val_scaled, test_scaled, scaler = scale_data(train_df, val_df, test_df)
    
    X_train = create_sliding_window(train_scaled, window_size)
    X_val = create_sliding_window(val_scaled, window_size)
    X_test = create_sliding_window(test_scaled, window_size)
    
    return X_train, X_val, X_test, scaler