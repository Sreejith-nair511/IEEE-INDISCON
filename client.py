"""
Flower client implementation for Federated Learning
"""

import torch
import numpy as np
import flwr as fl
from model import AnomalyDetector, train_epoch, evaluate, create_model_and_optimizer
from utils import compute_metrics, calculate_class_weights
from config import *


class FlowerClient(fl.client.NumPyClient):
    """
    Flower client for federated learning
    """
    def __init__(self, client_id, X_train, y_train, X_test, y_test):
        self.client_id = client_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store training data for class weight calculation
        self.y_train = torch.FloatTensor(y_train)
        
        # Create data loaders
        from utils import get_data_loaders
        self.train_loader, self.test_loader = get_data_loaders(
            X_train, y_train, BATCH_SIZE, TEST_SPLIT
        )
        
        # Initialize model, optimizer, and criterion
        self.model, self.optimizer, self.criterion = create_model_and_optimizer(self.device)
        
        # Calculate class weights for imbalanced data
        class_weights = calculate_class_weights(y_train)
        self.criterion = torch.nn.BCELoss(weight=class_weights[1])  # Use weight for positive class
        
        # Store number of samples
        self.num_samples = len(X_train)
    
    def get_parameters(self, config):
        """
        Get model parameters as numpy arrays
        
        Args:
            config: Configuration dictionary
            
        Returns:
            list: Model parameters as numpy arrays
        """
        return [param.detach().cpu().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters):
        """
        Set model parameters from numpy arrays
        
        Args:
            parameters (list): Model parameters as numpy arrays
        """
        # Convert numpy arrays to PyTorch tensors
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """
        Train the model on local data
        
        Args:
            parameters (list): Global model parameters
            config: Configuration dictionary
            
        Returns:
            tuple: (updated_parameters, num_samples, metrics)
        """
        # Set global model parameters
        self.set_parameters(parameters)
        
        # Compute class weights from local data
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        classes = np.array([0, 1])
        if len(np.unique(self.y_train.numpy())) > 1:
            weights = compute_class_weight('balanced', 
                                            classes=classes, 
                                            y=self.y_train.numpy())
            pos_weight = torch.tensor([weights[1]/weights[0]], dtype=torch.float32)
        else:
            pos_weight = torch.tensor([1.0])
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Train for LOCAL_EPOCHS
        train_loss = 0.0
        for epoch in range(LOCAL_EPOCHS):
            epoch_loss = train_epoch(
                self.model, self.train_loader, self.optimizer, criterion, self.device
            )
            train_loss += epoch_loss
            print(f"Client {self.client_id} - Epoch {epoch + 1}/{LOCAL_EPOCHS}, Loss: {epoch_loss:.4f}")
        
        avg_train_loss = train_loss / LOCAL_EPOCHS
        
        # Return updated parameters and metrics
        updated_params = self.get_parameters(config)
        metrics = {
            'loss': avg_train_loss,
            'client_id': self.client_id
        }
        
        return updated_params, self.num_samples, metrics
    
    def evaluate(self, parameters, config):
        """
        Evaluate the model on local test data
        
        Args:
            parameters (list): Global model parameters
            config: Configuration dictionary
            
        Returns:
            tuple: (loss, num_samples, metrics)
        """
        # Set global model parameters
        self.set_parameters(parameters)
        
        # Evaluate on test data
        test_loss, y_true, y_pred = evaluate(
            self.model, self.test_loader, self.criterion, self.device
        )
        
        # Compute metrics
        metrics = compute_metrics(y_true, y_pred)
        metrics['client_id'] = self.client_id
        
        print(f"Client {self.client_id} - Test Loss: {test_loss:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
        return test_loss, self.num_samples, metrics


def client_fn(client_id, client_data_list):
    """
    Factory function to create Flower clients
    
    Args:
        client_id (int): Client ID
        client_data_list (list): List of (X, y) tuples for each client
        
    Returns:
        FlowerClient: Configured Flower client
    """
    X_train, y_train, X_test, y_test = client_data_list[client_id]
    return FlowerClient(client_id, X_train, y_train, X_test, y_test)
