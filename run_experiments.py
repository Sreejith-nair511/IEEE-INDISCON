"""
Main orchestration script for running all experiments
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime

# Import all modules
from config import *
from utils import set_seed, generate_synthetic_data, non_iid_split, get_data_loaders
from centralized import run_centralized
from local_only import run_local_only
from server import run_federated, save_federated_metrics
from plots import generate_all_plots


def prepare_client_data(client_data_list):
    """
    Prepare client data with train/test splits
    
    Args:
        client_data_list (list): List of (X, y) tuples for each client
        
    Returns:
        list: List of (X_train, y_train, X_test, y_test) tuples
    """
    prepared_data = []
    
    for client_id, (X_client, y_client) in enumerate(client_data_list):
        train_loader, test_loader = get_data_loaders(X_client, y_client, BATCH_SIZE, TEST_SPLIT)
        
        # Extract data from loaders
        X_train = []
        y_train = []
        for batch_X, batch_y in train_loader:
            X_train.append(batch_X.numpy())
            y_train.append(batch_y.numpy())
        X_train = np.vstack(X_train)
        y_train = np.vstack(y_train).flatten()
        
        X_test = []
        y_test = []
        for batch_X, batch_y in test_loader:
            X_test.append(batch_X.numpy())
            y_test.append(batch_y.numpy())
        X_test = np.vstack(X_test)
        y_test = np.vstack(y_test).flatten()
        
        prepared_data.append((X_train, y_train, X_test, y_test))
    
    return prepared_data


def generate_benchmark_report(centralized_metrics, federated_metrics, local_metrics):
    """
    Generate and save benchmark report
    
    Args:
        centralized_metrics (dict): Centralized training metrics
        federated_metrics (dict): Federated learning metrics
        local_metrics (dict): Local-only training metrics
    """
    print("\n" + "="*80)
    print("BENCHMARK REPORT")
    print("="*80)
    
    # Create benchmark table
    benchmark_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Comm. Cost (MB)', 'Rounds'],
        'Centralized': [
            f"{centralized_metrics['accuracy']:.4f}",
            f"{centralized_metrics['precision']:.4f}",
            f"{centralized_metrics['recall']:.4f}",
            f"{centralized_metrics['f1']:.4f}",
            "N/A",
            "N/A"
        ],
        'Federated': [
            f"{federated_metrics['accuracy']:.4f}",
            f"{federated_metrics['precision']:.4f}",
            f"{federated_metrics['recall']:.4f}",
            f"{federated_metrics['f1']:.4f}",
            f"{federated_metrics.get('communication_cost_mb', 0):.2f}",
            f"{federated_metrics.get('num_rounds', 0)}"
        ],
        'Local-Only': [
            f"{local_metrics['accuracy']:.4f}",
            f"{local_metrics['precision']:.4f}",
            f"{local_metrics['recall']:.4f}",
            f"{local_metrics['f1']:.4f}",
            "N/A",
            "N/A"
        ]
    }
    
    # Print table
    df = pd.DataFrame(benchmark_data)
    print(df.to_string(index=False))
    
    # Validate federated performance (should be within 5% of centralized)
    centralized_f1 = centralized_metrics['f1']
    federated_f1 = federated_metrics['f1']
    performance_diff = abs(centralized_f1 - federated_f1) / centralized_f1 * 100
    
    print(f"\nPerformance Analysis:")
    print(f"Federated F1-Score: {federated_f1:.4f}")
    print(f"Centralized F1-Score: {centralized_f1:.4f}")
    print(f"Performance Difference: {performance_diff:.2f}%")
    
    if performance_diff <= 5.0:
        print("✅ Federated learning performance is within 5% of centralized baseline")
    else:
        print("⚠️  Federated learning performance differs by more than 5% from centralized baseline")
    
    # Save benchmark report
    report_content = f"""
Federated Learning-Based Distributed Anomaly Detection - Benchmark Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{df.to_string(index=False)}

Performance Analysis:
- Federated F1-Score: {federated_f1:.4f}
- Centralized F1-Score: {centralized_f1:.4f}
- Performance Difference: {performance_diff:.2f}%

Configuration:
- Number of Clients: {NUM_CLIENTS}
- Number of Rounds: {NUM_ROUNDS}
- Local Epochs: {LOCAL_EPOCHS}
- Batch Size: {BATCH_SIZE}
- Learning Rate: {LEARNING_RATE}
- Data Samples: {DATA_SAMPLES}
- Anomaly Ratio: {ANOMALY_RATIO}
"""
    
    report_path = os.path.join(METRICS_DIR, "benchmark_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"\nBenchmark report saved to {report_path}")
    
    return benchmark_data


def main():
    """
    Main function to run all experiments
    """
    print("="*80)
    print("FEDERATED LEARNING-BASED DISTRIBUTED ANOMALY DETECTION")
    print("IEEE Research Paper Implementation")
    print("="*80)
    
    # Set random seed for reproducibility
    set_seed(SEED)
    
    print("=" * 80)
    print("FEDERATED LEARNING-BASED DISTRIBUTED ANOMALY DETECTION")
    print("IEEE Research Paper Implementation")
    print("=" * 80)
    
    # Step 1: Generate synthetic dataset
    print(f"\nStep 1: Generating synthetic dataset...")
    print(f"Training Samples: {DATA_SAMPLES}, Features: {INPUT_DIM}, Anomaly Ratio: {ANOMALY_RATIO}")
    
    X_train_pool, y_train_pool, X_global_test, y_global_test = generate_synthetic_data(DATA_SAMPLES, INPUT_DIM, ANOMALY_RATIO, SEED)
    print(f"Training pool generated: {len(X_train_pool)} samples")
    print(f"Global test set generated: {len(X_global_test)} samples")
    print(f"Training class distribution - Normal: {np.sum(y_train_pool == 0)}, Anomaly: {np.sum(y_train_pool == 1)}")
    print(f"Test class distribution - Normal: {np.sum(y_global_test == 0)}, Anomaly: {np.sum(y_global_test == 1)}")
    
    # Step 2: Split data across clients (non-IID)
    print(f"\nStep 2: Splitting training pool across {NUM_CLIENTS} clients (Non-IID)...")
    client_data_list = non_iid_split(X_train_pool, y_train_pool, NUM_CLIENTS)
    
    for i, (X_client, y_client) in enumerate(client_data_list):
        anomaly_ratio = np.sum(y_client == 1) / len(y_client)
        print(f"Client {i}: {len(X_client)} samples, Anomaly Ratio: {anomaly_ratio:.3f}")
    
    # Step 3: Prepare global test set
    print(f"\nStep 3: Using global test set...")
    print(f"Global test set: {len(X_global_test)} samples (held-out, never seen by clients)")
    
    # Step 4: Run centralized baseline
    print(f"\nStep 4: Running centralized baseline...")
    centralized_metrics = run_centralized(X_train_pool, y_train_pool, X_global_test, y_global_test)
    
    # Step 5: Run local-only baseline
    print(f"\nStep 5: Running local-only baseline...")
    local_metrics = run_local_only(client_data_list)
    
    # Step 6: Run federated learning
    print(f"\nStep 6: Running federated learning...")
    prepared_client_data = prepare_client_data(client_data_list)
    history, round_metrics = run_federated(prepared_client_data, X_global_test, y_global_test)
    federated_metrics = save_federated_metrics(round_metrics)
    
    # Step 7: Generate plots
    print(f"\nStep 7: Generating plots...")
    generate_all_plots(round_metrics, centralized_metrics, federated_metrics, local_metrics, client_data_list)
    
    # Step 8: Generate benchmark report
    print(f"\nStep 8: Generating benchmark report...")
    benchmark_data = generate_benchmark_report(centralized_metrics, federated_metrics, local_metrics)
    
    print(f"\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Results saved in:")
    print(f"- Models: models/")
    print(f"- Plots: {PLOTS_DIR}")
    print(f"- Metrics: {METRICS_DIR}")
    print(f"- Benchmark Report: {METRICS_DIR}/benchmark_report.txt")
    
    return {
        'centralized': centralized_metrics,
        'federated': federated_metrics,
        'local_only': local_metrics,
        'benchmark': benchmark_data
    }


if __name__ == "__main__":
    results = main()
