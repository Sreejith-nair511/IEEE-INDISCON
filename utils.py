"""
Utility functions for Federated Learning-Based Distributed Anomaly Detection
"""

import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import os
from config import *


def set_seed(seed):
    """
    Set random seeds for reproducibility
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_synthetic_data(n_samples=5000, n_features=20, anomaly_ratio=0.2, seed=42):
    """
    Generate synthetic anomaly detection dataset with separate training pool and global test set
    
    Args:
        n_samples (int): Number of training samples to generate
        n_features (int): Number of features
        anomaly_ratio (float): Ratio of anomalies in the dataset
        seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train_pool, y_train_pool, X_global_test, y_global_test)
    """
    np.random.seed(seed)
    
    # Generate total samples (training pool + global test set)
    n_train_pool = n_samples
    n_global_test = 1000  # Fixed 1000 samples for global test set
    total_samples = n_train_pool + n_global_test
    
    # Calculate anomaly counts
    n_anomaly = int(total_samples * anomaly_ratio)
    n_normal = total_samples - n_anomaly
    
    # Normal data: Gaussian distribution
    X_normal = np.random.multivariate_normal(
        mean=[0] * n_features,
        cov=np.eye(n_features),
        size=n_normal
    )
    y_normal = np.zeros(n_normal)
    
    # Anomaly data: Slightly different distribution (more challenging)
    X_anomaly = np.random.multivariate_normal(
        mean=[0.8] * n_features,  # Reduced separation from 2.0 to 0.8
        cov=np.eye(n_features) * 1.2,  # Added variance
        size=n_anomaly
    )
    y_anomaly = np.ones(n_anomaly)
    
    # Combine and shuffle
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([y_normal, y_anomaly])
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split into training pool and global test set (stratified)
    from sklearn.model_selection import train_test_split
    X_train_pool, X_global_test, y_train_pool, y_global_test = train_test_split(
        X, y, test_size=n_global_test, stratify=y, random_state=seed
    )
    
    # Normalize features (fit on training pool only)
    scaler = StandardScaler()
    X_train_pool = scaler.fit_transform(X_train_pool)
    X_global_test = scaler.transform(X_global_test)
    
    return X_train_pool, y_train_pool, X_global_test, y_global_test


def non_iid_split(X, y, num_clients=5):
    """
    Split data across clients in non-IID fashion.
    Every client has BOTH classes, but with different ratios.
    """
    np.random.seed(42)
    
    # Separate by class
    normal_idx = np.where(y == 0)[0]
    anomaly_idx = np.where(y == 1)[0]
    
    np.random.shuffle(normal_idx)
    np.random.shuffle(anomaly_idx)
    
    # Target anomaly ratios per client
    anomaly_ratios = [0.20, 0.80, 0.40, 0.50, 0.30]
    
    client_data = []
    samples_per_client = len(y) // num_clients
    
    for i in range(num_clients):
        ratio = anomaly_ratios[i]
        n_anomaly = int(samples_per_client * ratio)
        n_normal = samples_per_client - n_anomaly
        
        # Ensure we don't exceed available samples
        n_anomaly = min(n_anomaly, len(anomaly_idx) // num_clients + 
                       len(anomaly_idx) % num_clients)
        n_normal = min(n_normal, len(normal_idx) // num_clients + 
                      len(normal_idx) % num_clients)
        
        # Take slices
        start_a = i * (len(anomaly_idx) // num_clients)
        end_a = start_a + n_anomaly
        start_n = i * (len(normal_idx) // num_clients)
        end_n = start_n + n_normal
        
        idx = np.concatenate([
            anomaly_idx[start_a:end_a],
            normal_idx[start_n:end_n]
        ])
        np.random.shuffle(idx)
        
        client_data.append((X[idx], y[idx]))
    
    return client_data


def get_data_loaders(X, y, batch_size, test_split):
    """
    Create PyTorch DataLoaders for training and testing
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Label vector
        batch_size (int): Batch size for DataLoaders
        test_split (float): Fraction of data to use for testing
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Handle edge case where test_split=0.0 (use all data for test)
    if test_split == 0.0:
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=SEED, stratify=y
        )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics
    
    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1
    """
    metrics = {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1': round(f1_score(y_true, y_pred, zero_division=0), 4)
    }
    return metrics


def save_metrics_csv(metrics_dict, filename):
    """
    Save metrics dictionary to CSV file
    
    Args:
        metrics_dict (dict): Dictionary containing metrics
        filename (str): Name of the CSV file
    """
    os.makedirs(METRICS_DIR, exist_ok=True)
    filepath = os.path.join(METRICS_DIR, filename)
    
    df = pd.DataFrame([metrics_dict])
    df.to_csv(filepath, index=False)
    print(f"Metrics saved to {filepath}")


def calculate_class_weights(y):
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        y (np.array): Label vector
        
    Returns:
        torch.Tensor: Class weights for both classes (0 and 1)
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Ensure we have both classes represented
    classes = np.array([0, 1])
    
    try:
        weights = compute_class_weight('balanced', classes=classes, y=y)
    except ValueError:
        # If one class is missing, use equal weights
        weights = np.array([1.0, 1.0])
    
    return torch.FloatTensor(weights)
