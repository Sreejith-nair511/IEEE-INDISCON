"""
Configuration file for Federated Learning-Based Distributed Anomaly Detection
"""

# Federated Learning Hyperparameters
NUM_CLIENTS = 5
NUM_ROUNDS = 30
LOCAL_EPOCHS = 10

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0005

# Model Architecture
INPUT_DIM = 20
HIDDEN1 = 64
HIDDEN2 = 32
OUTPUT_DIM = 1

# Data Generation
SEED = 42
DATA_SAMPLES = 5000
ANOMALY_RATIO = 0.2
TEST_SPLIT = 0.2

# Directory Paths
RESULTS_DIR = "results/"
PLOTS_DIR = "results/plots/"
METRICS_DIR = "results/metrics/"
