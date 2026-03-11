import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from typing import Tuple

class AnomalyDetector:
    def __init__(self, nu: float = 0.025, gamma: float = 100.0):
        self.pca = PCA(n_components=1)
        self.ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
        
    def fit(self, train_errors: np.ndarray):
        pca_features = self.pca.fit_transform(train_errors)
        
        self.ocsvm.fit(pca_features)
        
    def predict(self, test_errors: np.ndarray) -> np.ndarray:
        pca_features = self.pca.transform(test_errors)
        
        # OCSVM returns 1 for normal, -1 for anomaly. 
        # Map to 0 (normal) and 1 (anomaly/disrupted) for easier evaluation
        preds = self.ocsvm.predict(pca_features)
        anomalies = np.where(preds == -1, 1, 0)
        
        return anomalies