"""
Federated Learning server implementation using Flower
"""

import torch
import numpy as np
import flwr as fl
from typing import List, Tuple, Dict
from model import AnomalyDetector, evaluate, create_model_and_optimizer, count_parameters
from utils import compute_metrics, get_data_loaders, calculate_class_weights
from config import *


# Global list to track round metrics
round_metrics = []


def create_federated_strategy(num_rounds=NUM_ROUNDS):
    """
    Create and configure federated learning strategy
    
    Args:
        num_rounds (int): Number of federated learning rounds
        
    Returns:
        fl.server.strategy.FedAvg: Configured FedAvg strategy
    """
    
    def on_fit_config_fn(round_num: int) -> Dict:
        """
        Configuration function for training rounds
        
        Args:
            round_num (int): Current round number
            
        Returns:
            Dict: Configuration for the round
        """
        config = {
            "round": round_num,
            "local_epochs": LOCAL_EPOCHS,
        }
        return config
    
    def evaluate_fn(server_round: int, parameters: List[np.ndarray], config: Dict):
        """
        Global evaluation function using a held-out test set
        
        Args:
            server_round (int): Current round number
            parameters (List[np.ndarray]): Global model parameters
            config (Dict): Configuration dictionary
            
        Returns:
            Tuple[float, int, Dict]: Loss, number of samples, and metrics
        """
        # This function will be called with global test data
        # We'll set the global test data in run_federated function
        if not hasattr(evaluate_fn, 'global_test_loader') or not hasattr(evaluate_fn, 'device'):
            return 0.0, 1, {"accuracy": 0.0}
        
        # Create model and set global parameters
        model, _, criterion = create_model_and_optimizer(evaluate_fn.device)
        
        # Set parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        
        # Evaluate on global test set
        test_loss, y_true, y_pred = evaluate(
            model, evaluate_fn.global_test_loader, criterion, evaluate_fn.device
        )
        
        # Compute metrics
        metrics = compute_metrics(y_true, y_pred)
        
        # Store round metrics
        round_metrics.append({
            'round': server_round,
            'loss': test_loss,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })
        
        print(f"Round {server_round} - Global Test Loss: {test_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        return test_loss, len(evaluate_fn.global_test_loader.dataset), metrics
    
    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        on_fit_config_fn=on_fit_config_fn,
        evaluate_fn=evaluate_fn,
        initial_parameters=None,  # Will be set from first client
    )
    
    return strategy


def run_federated_simple(client_data_list, global_test_X, global_test_y):
    """
    Run simple federated learning simulation without ray
    
    Args:
        client_data_list (list): List of client data tuples
        global_test_X (np.array): Global test features
        global_test_y (np.array): Global test labels
        
    Returns:
        tuple: (round_metrics, final_metrics)
    """
    print("Starting Simple Federated Learning Simulation...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate independent global test set (1000 samples, different seed)
    from sklearn.model_selection import train_test_split
    from utils import generate_synthetic_data
    
    # Global test set must be 1000 samples, completely separate from client data
    # Generate it independently with a different seed
    X_global_test, y_global_test, _, _ = generate_synthetic_data(
        n_samples=1000, 
        n_features=20, 
        anomaly_ratio=0.2, 
        seed=99          # different seed from training data (42)
    )
    
    print(f"Global test set generated: {len(X_global_test)} samples")
    print(f"Global test class distribution - Normal: {np.sum(y_global_test == 0)}, Anomaly: {np.sum(y_global_test == 1)}")
    
    # Create global test data loader - use proper held-out test set
    _, global_test_loader = get_data_loaders(
        X_global_test, y_global_test, BATCH_SIZE, 0.0
    )
    
    # Initialize global model
    from model import AnomalyDetector, create_model_and_optimizer, evaluate, get_model_weights, set_model_weights
    global_model, _, criterion = create_model_and_optimizer(device)
    
    # Calculate class weights for global test set
    from utils import calculate_class_weights, compute_metrics
    class_weights = calculate_class_weights(global_test_y)
    pos_weight = class_weights[1] / class_weights[0]
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    
    # Create clients
    from client import FlowerClient
    clients = []
    for client_id, (X_train, y_train, X_test, y_test) in enumerate(client_data_list):
        client = FlowerClient(client_id, X_train, y_train, X_test, y_test)
        clients.append(client)
    
    # Federated learning rounds
    round_metrics = []
    
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n--- Round {round_num}/{NUM_ROUNDS} ---")
        
        # Get current global parameters
        global_params = get_model_weights(global_model)
        
        # Train each client
        client_params = []
        client_sizes = []
        client_metrics = []
        
        for client in clients:
            # Train client
            params, num_samples, metrics = client.fit(global_params, {})
            client_params.append(params)
            client_sizes.append(num_samples)
            client_metrics.append(metrics)
            print(f"Client {metrics['client_id']}: Loss = {metrics['loss']:.4f}")
        
        # Federated averaging (weighted by client data size)
        total_samples = sum(client_sizes)
        averaged_params = []
        
        for param_idx in range(len(global_params)):
            weighted_param = np.zeros_like(global_params[param_idx])
            for client_param, client_size in zip(client_params, client_sizes):
                weight = client_size / total_samples
                weighted_param += weight * client_param[param_idx]
            averaged_params.append(weighted_param)
        
        # Update global model
        set_model_weights(global_model, averaged_params)
        
        # Evaluate global model
        test_loss, y_true, y_pred = evaluate(global_model, global_test_loader, criterion, device)
        metrics = compute_metrics(y_true, y_pred)
        
        round_metric = {
            'round': round_num,
            'loss': test_loss,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        }
        round_metrics.append(round_metric)
        
        print(f"Round {round_num} - Global Test Loss: {test_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    
    print("Simple Federated Learning Simulation Completed!")
    
    # Calculate final metrics
    final_metrics = round_metrics[-1] if round_metrics else {}
    final_metrics['communication_cost_mb'] = calculate_communication_cost()
    final_metrics['num_rounds'] = NUM_ROUNDS
    
    return round_metrics, final_metrics


def run_federated(client_data_list, global_test_X, global_test_y):
    """
    Run federated learning simulation (wrapper for simple implementation)
    
    Args:
        client_data_list (list): List of client data tuples
        global_test_X (np.array): Global test features
        global_test_y (np.array): Global test labels
        
    Returns:
        tuple: (history, round_metrics)
    """
    # Use simple implementation to avoid ray dependency
    round_metrics, final_metrics = run_federated_simple(client_data_list, global_test_X, global_test_y)
    return None, round_metrics


def calculate_communication_cost():
    """
    Calculate communication cost for federated learning
    
    Returns:
        float: Communication cost in MB
    """
    # Create a dummy model to count parameters
    model = AnomalyDetector()
    param_count = count_parameters(model)
    
    # Communication cost = (model_param_count * 4 bytes * 2 * num_rounds * num_clients) / 1e6 MB
    # *2 for upload + download
    comm_cost = (param_count * 4 * 2 * NUM_ROUNDS * NUM_CLIENTS) / 1e6
    
    return comm_cost


def save_federated_metrics(round_metrics, filename="federated_metrics.csv"):
    """
    Save federated learning metrics to CSV
    
    Args:
        round_metrics (list): List of round-wise metrics
        filename (str): Output filename
    """
    import pandas as pd
    import os
    
    os.makedirs(METRICS_DIR, exist_ok=True)
    filepath = os.path.join(METRICS_DIR, filename)
    
    df = pd.DataFrame(round_metrics)
    df.to_csv(filepath, index=False)
    print(f"Federated metrics saved to {filepath}")
    
    # Save final metrics
    if round_metrics:
        final_metrics = round_metrics[-1]
        final_metrics['communication_cost_mb'] = calculate_communication_cost()
        final_metrics['num_rounds'] = NUM_ROUNDS
        
        final_filepath = os.path.join(METRICS_DIR, "federated_final_metrics.csv")
        pd.DataFrame([final_metrics]).to_csv(final_filepath, index=False)
        print(f"Final federated metrics saved to {final_filepath}")
    
    return final_metrics if round_metrics else {}
