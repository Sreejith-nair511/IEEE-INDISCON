# Federated Learning-Based Distributed Anomaly Detection in Smart Infrastructure Networks

This repository contains a complete, production-ready implementation of the IEEE research paper titled "Federated Learning-Based Distributed Anomaly Detection in Smart Infrastructure Networks". The implementation demonstrates federated learning for anomaly detection in smart infrastructure scenarios with non-IID data distribution across multiple clients.

## 🏗️ Project Structure

```
federated_anomaly/
├── data/                    # Generated datasets (auto-created)
├── models/                  # Saved model checkpoints (auto-created)
├── results/
│   ├── plots/              # Generated visualizations (auto-created)
│   └── metrics/            # CSV metrics and reports (auto-created)
├── config.py               # Central configuration file
├── utils.py                # Utility functions for data generation and metrics
├── model.py                # PyTorch AnomalyDetector model definition
├── client.py               # Flower client implementation
├── server.py               # Federated learning server with FedAvg strategy
├── centralized.py          # Centralized training baseline
├── local_only.py           # Local-only training baseline
├── run_experiments.py      # Main orchestration script
├── plots.py                # Visualization functions
└── README.md               # This file
```

## 🚀 Features

- **Federated Learning**: Uses Flower framework with FedAvg strategy
- **Non-IID Data Distribution**: Realistic client data splitting with varying anomaly ratios
- **Multiple Baselines**: Compares federated learning with centralized and local-only approaches
- **Comprehensive Evaluation**: Tracks accuracy, precision, recall, F1-score, and communication costs
- **Visualization**: Generates publication-quality plots for results analysis
- **Reproducible**: Fixed random seeds and deterministic training

## 📋 Requirements

- Python 3.8+
- PyTorch
- Flower (flwr) >= 1.0.0
- scikit-learn
- matplotlib
- numpy
- pandas
- tqdm

## 🔧 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd federated_anomaly
```

2. Install the required packages:
```bash
pip install torch flwr scikit-learn matplotlib numpy pandas tqdm
```

3. (Optional) Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch flwr scikit-learn matplotlib numpy pandas tqdm
```

## 🏃‍♂️ How to Run

Execute the main experiment script:

```bash
python run_experiments.py
```

This will automatically:
1. Generate synthetic anomaly detection dataset
2. Split data across 5 clients in non-IID fashion
3. Run centralized training baseline
4. Run local-only training baseline
5. Run federated learning simulation
6. Generate all visualization plots
7. Save comprehensive metrics and benchmark report

## 📊 Expected Outputs

After running the experiments, you will find:

### Models (`models/`)
- `centralized_model.pth` - Trained centralized model
- `local_client_{i}_model.pth` - Individual client models (i=0-4)

### Plots (`results/plots/`)
- `federated_accuracy_per_round.png` - Federated accuracy over communication rounds
- `federated_loss_per_round.png` - Federated loss over communication rounds
- `comparison_bar_chart.png` - Performance comparison across approaches
- `client_distribution.png` - Non-IID data distribution visualization

### Metrics (`results/metrics/`)
- `centralized_metrics.csv` - Centralized training results
- `federated_metrics.csv` - Round-wise federated metrics
- `federated_final_metrics.csv` - Final federated results
- `local_only_avg_metrics.csv` - Local-only average results
- `local_client_{i}_metrics.csv` - Individual client results
- `benchmark_report.txt` - Comprehensive benchmark comparison

## ⚙️ Configuration

Key hyperparameters can be modified in `config.py`:

```python
# Federated Learning Hyperparameters
NUM_CLIENTS = 5
NUM_ROUNDS = 20
LOCAL_EPOCHS = 5

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001

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
```

## 🧠 Model Architecture

The AnomalyDetector is a feedforward neural network:
- Input Layer → Linear(20, 64) → ReLU → Dropout(0.3)
- Hidden Layer → Linear(64, 32) → ReLU → Dropout(0.2)
- Output Layer → Linear(32, 1) → Sigmoid

## 📈 Non-IID Data Distribution

The implementation creates realistic non-IID data distribution:
- **Client 0**: 80% normal samples, 20% anomalies
- **Client 1**: 20% normal samples, 80% anomalies  
- **Clients 2-4**: Mixed stratified distribution

## 🎯 Performance Validation

The federated learning model is designed to achieve F1-score within 5% of the centralized baseline while providing privacy benefits and reduced communication overhead.

## 📋 Benchmark Report

The system generates a comprehensive benchmark report comparing:

| Metric     | Centralized | Federated | Local-Only |
|------------|-------------|-----------|------------|
| Accuracy   | X.XXXX      | X.XXXX    | X.XXXX     |
| Precision  | X.XXXX      | X.XXXX    | X.XXXX     |
| Recall     | X.XXXX      | X.XXXX    | X.XXXX     |
| F1 Score   | X.XXXX      | X.XXXX    | X.XXXX     |
| Comm. Cost | N/A         | XX.XX MB  | N/A        |
| Rounds     | N/A         | 20        | N/A        |

## 🔬 Research Paper Reference

This implementation is based on the IEEE research paper:
**"Federated Learning-Based Distributed Anomaly Detection in Smart Infrastructure Networks"**

The paper presents a novel approach for detecting anomalies in smart infrastructure networks using federated learning, enabling privacy-preserving collaborative learning while maintaining high detection accuracy.

## 🛠️ Key Components

### Client Implementation (`client.py`)
- FlowerClient class inheriting from `flwr.client.NumPyClient`
- Local training with weighted BCE loss for class imbalance
- Parameter synchronization with federated server

### Server Implementation (`server.py`)
- FedAvg strategy with custom evaluation function
- Global test set evaluation
- Communication cost tracking

### Data Generation (`utils.py`)
- Synthetic data generation using scikit-learn
- Non-IID splitting across clients
- Comprehensive metrics computation

### Visualization (`plots.py`)
- Publication-quality matplotlib plots
- Federated learning progress tracking
- Performance comparison charts

## 🐛 Troubleshooting

### Common Issues

1. **Flower Installation**: Ensure you have flwr >= 1.0.0
   ```bash
   pip install flwr>=1.0.0
   ```

2. **CUDA Issues**: If you encounter CUDA errors, the system will automatically fall back to CPU

3. **Memory Issues**: Reduce `DATA_SAMPLES` or `BATCH_SIZE` if you run out of memory

4. **Reproducibility**: All experiments use fixed seeds. Results should be consistent across runs.

## 📝 License

This implementation is provided for research and educational purposes. Please cite the original IEEE paper if you use this code in your research.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📧 Contact

For questions about this implementation, please refer to the original IEEE research paper or open an issue in the repository.
