"""
Local-only training baseline implementation
"""

import torch
import numpy as np
from model import AnomalyDetector, train_epoch, evaluate, create_model_and_optimizer
from utils import compute_metrics, get_data_loaders, calculate_class_weights, save_metrics_csv
from config import *


def run_local_only(client_data_list):
    """
    Run local-only training on each client independently
    
    Args:
        client_data_list (list): List of (X, y) tuples for each client
        
    Returns:
        dict: Dictionary containing average metrics across all clients
    """
    print("Starting Local-Only Training...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Calculate total epochs (LOCAL_EPOCHS * NUM_ROUNDS)
    total_epochs = LOCAL_EPOCHS * NUM_ROUNDS
    
    all_client_metrics = []
    
    # Train each client independently
    for client_id, (X_client, y_client) in enumerate(client_data_list):
        print(f"\nTraining Client {client_id}...")
        
        # Create data loaders - use 20% held-out test split for each client
        train_loader, test_loader = get_data_loaders(X_client, y_client, BATCH_SIZE, 0.2)
        
        print(f"Client {client_id} training set: {len(train_loader.dataset)} samples")
        print(f"Client {client_id} test set: {len(test_loader.dataset)} samples")
        
        # Create model, optimizer, and criterion
        model, optimizer, criterion = create_model_and_optimizer(device)
        
        # Calculate class weights for imbalanced data
        class_weights = calculate_class_weights(y_client)
        pos_weight = class_weights[1] / class_weights[0]
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        
        # Training loop
        train_losses = []
        print(f"Training Client {client_id} for {total_epochs} epochs...")
        
        for epoch in range(total_epochs):
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            train_losses.append(train_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Client {client_id} - Epoch {epoch + 1}/{total_epochs}, Loss: {train_loss:.4f}")
        
        # Evaluate on client's test set
        print(f"Evaluating Client {client_id}...")
        test_loss, y_true, y_pred = evaluate(
            model, test_loader, criterion, device
        )
        
        # Compute metrics
        client_metrics = compute_metrics(y_true, y_pred)
        client_metrics['test_loss'] = round(test_loss, 4)
        client_metrics['client_id'] = client_id
        client_metrics['num_epochs'] = total_epochs
        client_metrics['communication_cost_mb'] = 0.0  # No communication in local-only
        client_metrics['num_rounds'] = 0  # No rounds in local-only
        
        print(f"Client {client_id} Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Accuracy: {client_metrics['accuracy']:.4f}")
        print(f"Precision: {client_metrics['precision']:.4f}")
        print(f"Recall: {client_metrics['recall']:.4f}")
        print(f"F1-Score: {client_metrics['f1']:.4f}")
        
        all_client_metrics.append(client_metrics)
        
        # Save client model
        model_path = f"models/local_client_{client_id}_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Client {client_id} model saved to {model_path}")
        
        # Save client metrics
        save_metrics_csv(client_metrics, f"local_client_{client_id}_metrics.csv")
        
        # Save client training history
        training_history = {
            'epoch': list(range(1, total_epochs + 1)),
            'train_loss': [round(loss, 4) for loss in train_losses]
        }
        save_metrics_csv(training_history, f"local_client_{client_id}_training_history.csv")
    
    # Calculate average metrics across all clients
    avg_metrics = {
        'accuracy': round(np.mean([m['accuracy'] for m in all_client_metrics]), 4),
        'precision': round(np.mean([m['precision'] for m in all_client_metrics]), 4),
        'recall': round(np.mean([m['recall'] for m in all_client_metrics]), 4),
        'f1': round(np.mean([m['f1'] for m in all_client_metrics]), 4),
        'test_loss': round(np.mean([m['test_loss'] for m in all_client_metrics]), 4),
        'num_epochs': total_epochs,
        'communication_cost_mb': 0.0,
        'num_rounds': 0,
        'num_clients': NUM_CLIENTS
    }
    
    # Calculate standard deviations
    std_metrics = {
        'accuracy_std': round(np.std([m['accuracy'] for m in all_client_metrics]), 4),
        'precision_std': round(np.std([m['precision'] for m in all_client_metrics]), 4),
        'recall_std': round(np.std([m['recall'] for m in all_client_metrics]), 4),
        'f1_std': round(np.std([m['f1'] for m in all_client_metrics]), 4),
        'test_loss_std': round(np.std([m['test_loss'] for m in all_client_metrics]), 4)
    }
    
    # Combine average and std metrics
    final_metrics = {**avg_metrics, **std_metrics}
    
    print(f"\nLocal-Only Training Results (Average across {NUM_CLIENTS} clients):")
    print(f"Test Loss: {avg_metrics['test_loss']:.4f} ± {std_metrics['test_loss_std']:.4f}")
    print(f"Accuracy: {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy_std']:.4f}")
    print(f"Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision_std']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f} ± {std_metrics['recall_std']:.4f}")
    print(f"F1-Score: {avg_metrics['f1']:.4f} ± {std_metrics['f1_std']:.4f}")
    
    # Save average metrics
    save_metrics_csv(final_metrics, "local_only_avg_metrics.csv")
    
    # Save all client metrics
    import pandas as pd
    import os
    all_metrics_df = pd.DataFrame(all_client_metrics)
    all_metrics_path = os.path.join("results/metrics", "local_only_all_clients.csv")
    all_metrics_df.to_csv(all_metrics_path, index=False)
    print(f"All client metrics saved to {all_metrics_path}")
    
    print("Local-Only Training Completed!")
    
    return final_metrics
