"""
Visualization functions for generating plots and saving results
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from config import *


def set_plot_style():
    """Set consistent plot style"""
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3


def plot_federated_accuracy_per_round(round_metrics):
    """
    Plot federated learning accuracy per round
    
    Args:
        round_metrics (list): List of round-wise metrics
    """
    set_plot_style()
    
    rounds = [m['round'] for m in round_metrics]
    accuracies = [m['accuracy'] for m in round_metrics]
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accuracies, 'b-o', linewidth=2, markersize=6, markerfacecolor='blue', markeredgecolor='blue')
    plt.xlabel('Communication Round')
    plt.ylabel('Global Accuracy')
    plt.title('Federated Learning: Accuracy vs Communication Rounds')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, max(rounds) + 1))
    plt.ylim(0, 1)
    
    # Add value labels on points
    for i, (round_num, acc) in enumerate(zip(rounds, accuracies)):
        plt.annotate(f'{acc:.3f}', (round_num, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    filepath = os.path.join(PLOTS_DIR, "federated_accuracy_per_round.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Federated accuracy plot saved to {filepath}")


def plot_federated_loss_per_round(round_metrics):
    """
    Plot federated learning loss per round
    
    Args:
        round_metrics (list): List of round-wise metrics
    """
    set_plot_style()
    
    rounds = [m['round'] for m in round_metrics]
    losses = [m['loss'] for m in round_metrics]
    
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, losses, 'r-o', linewidth=2, markersize=6, markerfacecolor='red', markeredgecolor='red')
    plt.xlabel('Communication Round')
    plt.ylabel('Global Loss')
    plt.title('Federated Learning: Loss vs Communication Rounds')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, max(rounds) + 1))
    
    # Add value labels on points
    for i, (round_num, loss) in enumerate(zip(rounds, losses)):
        plt.annotate(f'{loss:.3f}', (round_num, loss), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    filepath = os.path.join(PLOTS_DIR, "federated_loss_per_round.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Federated loss plot saved to {filepath}")


def plot_comparison_bar_chart(centralized_metrics, federated_metrics, local_metrics):
    """
    Plot comparison bar chart between different approaches
    
    Args:
        centralized_metrics (dict): Centralized training metrics
        federated_metrics (dict): Federated learning metrics
        local_metrics (dict): Local-only training metrics
    """
    set_plot_style()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    centralized_values = [
        centralized_metrics['accuracy'],
        centralized_metrics['precision'],
        centralized_metrics['recall'],
        centralized_metrics['f1']
    ]
    federated_values = [
        federated_metrics['accuracy'],
        federated_metrics['precision'],
        federated_metrics['recall'],
        federated_metrics['f1']
    ]
    local_values = [
        local_metrics['accuracy'],
        local_metrics['precision'],
        local_metrics['recall'],
        local_metrics['f1']
    ]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars1 = ax.bar(x - width, centralized_values, width, label='Centralized', color='green', alpha=0.8)
    bars2 = ax.bar(x, federated_values, width, label='Federated', color='blue', alpha=0.8)
    bars3 = ax.bar(x + width, local_values, width, label='Local-Only', color='orange', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison: Centralized vs Federated vs Local-Only')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    filepath = os.path.join(PLOTS_DIR, "comparison_bar_chart.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison bar chart saved to {filepath}")


def plot_client_distribution(client_data_list):
    """
    Plot client data distribution showing anomaly ratios
    
    Args:
        client_data_list (list): List of (X, y) tuples for each client
    """
    set_plot_style()
    
    client_ids = []
    anomaly_ratios = []
    
    for i, (X_client, y_client) in enumerate(client_data_list):
        client_ids.append(f'Client {i}')
        anomaly_ratio = np.sum(y_client == 1) / len(y_client)
        anomaly_ratios.append(anomaly_ratio)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(client_ids, anomaly_ratios, color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1)
    
    plt.xlabel('Client ID')
    plt.ylabel('Anomaly Ratio')
    plt.title('Non-IID Data Distribution Across Clients')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, ratio in zip(bars, anomaly_ratios):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{ratio:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add horizontal line for overall anomaly ratio
    overall_ratio = ANOMALY_RATIO
    plt.axhline(y=overall_ratio, color='red', linestyle='--', alpha=0.7, label=f'Overall Ratio ({overall_ratio:.2f})')
    plt.legend()
    
    plt.tight_layout()
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    filepath = os.path.join(PLOTS_DIR, "client_distribution.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Client distribution plot saved to {filepath}")


def generate_all_plots(round_metrics, centralized_metrics, federated_metrics, local_metrics, client_data_list):
    """
    Generate all required plots
    
    Args:
        round_metrics (list): List of round-wise metrics
        centralized_metrics (dict): Centralized training metrics
        federated_metrics (dict): Federated learning metrics
        local_metrics (dict): Local-only training metrics
        client_data_list (list): List of client data
    """
    print("Generating visualization plots...")
    
    if round_metrics:
        plot_federated_accuracy_per_round(round_metrics)
        plot_federated_loss_per_round(round_metrics)
    
    plot_comparison_bar_chart(centralized_metrics, federated_metrics, local_metrics)
    plot_client_distribution(client_data_list)
    
    print("All plots generated successfully!")
    print(f"Plots saved in: {PLOTS_DIR}")
    
    # List generated plots
    plot_files = [
        "federated_accuracy_per_round.png",
        "federated_loss_per_round.png", 
        "comparison_bar_chart.png",
        "client_distribution.png"
    ]
    
    print("\nGenerated plots:")
    for plot_file in plot_files:
        filepath = os.path.join(PLOTS_DIR, plot_file)
        if os.path.exists(filepath):
            print(f"  ✅ {plot_file}")
        else:
            print(f"  ❌ {plot_file} (not found)")


if __name__ == "__main__":
    # This allows running plots.py independently for testing
    print("Plot generation module loaded successfully!")
