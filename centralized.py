"""
Centralized training baseline implementation
"""

import torch
import numpy as np
from model import AnomalyDetector, train_epoch, evaluate, create_model_and_optimizer
from utils import compute_metrics, get_data_loaders, calculate_class_weights, save_metrics_csv
from config import *


def run_centralized(X_train_pool, y_train_pool, X_global_test, y_global_test):
    """
    Run centralized training baseline
    
    Args:
        X_train_pool (np.array): Training feature matrix
        y_train_pool (np.array): Training label vector
        X_global_test (np.array): Global test feature matrix
        y_global_test (np.array): Global test label vector
        
    Returns:
        dict: Training and evaluation metrics
    """
    print("Starting Centralized Training...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders - use 80/20 split on training pool (5000 samples)
    train_loader, test_loader = get_data_loaders(X_train_pool, y_train_pool, BATCH_SIZE, 0.2)
    
    print(f"Centralized training set: {len(train_loader.dataset)} samples")
    print(f"Centralized test set: {len(test_loader.dataset)} samples")
    
    # Create model, optimizer, and criterion
    model, optimizer, criterion = create_model_and_optimizer(device)
    
    # Calculate class weights for imbalanced data
    class_weights = calculate_class_weights(y_train_pool)
    pos_weight = class_weights[1] / class_weights[0]
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    
    # Training loop
    num_epochs = 30
    train_losses = []
    
    print(f"Training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        train_losses.append(train_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, y_true, y_pred = evaluate(
        model, test_loader, criterion, device
    )
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    metrics['test_loss'] = round(test_loss, 4)
    metrics['num_epochs'] = num_epochs
    metrics['communication_cost_mb'] = 0.0  # No communication in centralized
    metrics['num_rounds'] = 0  # No rounds in centralized
    
    print(f"Centralized Training Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    # Save model
    model_path = "models/centralized_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    save_metrics_csv(metrics, "centralized_metrics.csv")
    
    # Save training history
    training_history = {
        'epoch': list(range(1, num_epochs + 1)),
        'train_loss': [round(loss, 4) for loss in train_losses]
    }
    save_metrics_csv(training_history, "centralized_training_history.csv")
    
    print("Centralized Training Completed!")
    
    return metrics
