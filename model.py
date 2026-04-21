"""
PyTorch model definition for Anomaly Detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from config import *


class AnomalyDetector(nn.Module):
    """
    Neural network model for anomaly detection
    """
    def __init__(self, input_dim=INPUT_DIM, hidden1=HIDDEN1, hidden2=HIDDEN2, output_dim=OUTPUT_DIM):
        super(AnomalyDetector, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)  # Raw logits, no sigmoid
    
    def predict(self, x):
        logits = self.forward(x)
        return (torch.sigmoid(logits) > 0.3).float()


def train_epoch(model, loader, optimizer, criterion, device='cpu'):
    """
    Train model for one epoch
    
    Args:
        model (nn.Module): PyTorch model
        loader (DataLoader): Training data loader
        optimizer (torch.optim): Optimizer
        criterion (nn.Module): Loss function
        device (str): Device to run on
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_X, batch_y in tqdm(loader, desc="Training", leave=False):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, loader, criterion, device='cpu'):
    """
    Evaluate model on test data
    
    Args:
        model (nn.Module): PyTorch model
        loader (DataLoader): Test data loader
        criterion (nn.Module): Loss function
        device (str): Device to run on
        
    Returns:
        tuple: (avg_loss, y_true, y_pred_binary)
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch).squeeze()
            y_batch = y_batch.squeeze()
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            num_batches += 1
            
            # Use 0.3 threshold for predictions
            preds = (torch.sigmoid(outputs) > 0.3).float()
            if preds.dim() == 0:
                all_preds.append(preds.cpu().item())
            else:
                all_preds.extend(preds.cpu().numpy())
            if y_batch.dim() == 0:
                all_labels.append(y_batch.cpu().item())
            else:
                all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    return avg_loss, np.array(all_labels), np.array(all_preds)


def get_model_weights(model):
    """
    Get model weights as numpy arrays
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        list: List of numpy arrays representing model weights
    """
    return [param.detach().cpu().numpy() for param in model.parameters()]


def set_model_weights(model, weights):
    """
    Set model weights from numpy arrays
    
    Args:
        model (nn.Module): PyTorch model
        weights (list): List of numpy arrays representing model weights
    """
    state_dict = model.state_dict()
    param_keys = list(state_dict.keys())
    
    for key, param in zip(param_keys, weights):
        state_dict[key] = torch.FloatTensor(param)
    
    model.load_state_dict(state_dict)


def create_model_and_optimizer(device='cpu'):
    """
    Create model, optimizer, and loss function
    
    Args:
        device (str): Device to run on
        
    Returns:
        tuple: (model, optimizer, criterion)
    """
    model = AnomalyDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    return model, optimizer, criterion


def count_parameters(model):
    """
    Count the number of trainable parameters in the model
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
