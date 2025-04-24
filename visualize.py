import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple


def setup_plot_style():
    """Set up the plot style for consistent visualizations."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18


def visualize_model_comparison(results_path: str = "evaluation_results.csv",
                              save_path: Optional[str] = "visualizations/model_comparison.png"):
    """
    Visualize model comparison from evaluation results.

    Args:
        results_path: Path to the evaluation results CSV file
        save_path: Path to save the visualization (if None, just displays)
    """
    # Create visualizations directory if it doesn't exist
    if save_path and not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # Set up plot style
    setup_plot_style()

    # Load results
    results = pd.read_csv(results_path)

    # Sort by accuracy
    results = results.sort_values(by="Accuracy", ascending=False)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Plot accuracy comparison
    models = results["Model"]
    accuracies = results["Accuracy"]
    std_devs = results["StdDev"]

    # Bar plot for accuracy
    bars = ax1.bar(models, accuracies, color='skyblue', alpha=0.7)
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(min(accuracies) - 0.01, 1.0)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate bars with accuracy values
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=0)

    # Bar plot for standard deviation
    bars = ax2.bar(models, std_devs, color='salmon', alpha=0.7)
    ax2.set_title('Model Standard Deviation')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Standard Deviation')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotate bars with std dev values
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=0)

    plt.tight_layout()

    # Save visualization
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Model comparison visualization saved to {save_path}")
    else:
        # If no save path provided, save to default location
        default_path = "visualizations/model_comparison.png"
        if not os.path.exists(os.path.dirname(default_path)):
            os.makedirs(os.path.dirname(default_path))
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"✅ Model comparison visualization saved to {default_path}")

    # Close the plot to avoid displaying it in non-interactive environments
    plt.close()


def visualize_prediction(prediction: Dict,
                        save_path: Optional[str] = None):
    """
    Visualize a single startup prediction.

    Args:
        prediction: Dictionary containing prediction results
        save_path: Path to save the visualization (if None, just displays)
    """
    # Create visualizations directory if it doesn't exist and save_path is provided
    if save_path and not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    # Set up plot style
    setup_plot_style()

    # Extract prediction data
    startup_name = prediction['startup_name']
    success_prob = prediction['success_probability']
    failure_prob = 100 - success_prob
    prediction_result = prediction['prediction']

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Gauge chart for success probability
    gauge_colors = ['#ff9999', '#ffcc99', '#ffff99', '#99ff99', '#66ff66']
    threshold_values = [20, 40, 60, 80, 100]

    # Determine color based on probability
    color_idx = next((i for i, v in enumerate(threshold_values) if success_prob <= v), 0)
    gauge_color = gauge_colors[color_idx]

    # Create a simple gauge-like visualization
    # Create a half-circle
    angles = np.linspace(0, np.pi, 100)
    ax1.plot(np.cos(angles), np.sin(angles), color='black', lw=2)

    # Add gauge ticks and labels
    for i, threshold in enumerate(threshold_values):
        angle = np.pi * (threshold / 100)
        x = np.cos(angle)
        y = np.sin(angle)
        ax1.plot([0, x], [0, y], color='black', lw=1, alpha=0.3)
        ax1.text(1.1 * x, 1.1 * y, f"{threshold}%",
                ha='center', va='center', fontsize=12)

    # Plot the gauge needle
    needle_angle = np.pi * (success_prob / 100)
    x = np.cos(needle_angle)
    y = np.sin(needle_angle)
    ax1.arrow(0, 0, 0.8 * x, 0.8 * y, head_width=0.05, head_length=0.1,
             fc='red', ec='red', lw=2)

    # Add colored gauge background
    for i in range(len(threshold_values)):
        start_angle = 0 if i == 0 else np.pi * (threshold_values[i-1] / 100)
        end_angle = np.pi * (threshold_values[i] / 100)

        angles = np.linspace(start_angle, end_angle, 50)
        xs = np.cos(angles)
        ys = np.sin(angles)

        ax1.fill_between(xs, 0, ys, color=gauge_colors[i], alpha=0.3)

    # Add probability value in center
    ax1.text(0, 0, f"{success_prob}%", ha='center', va='center',
            fontsize=18, fontweight='bold')

    ax1.set_title(f"Success Probability for {startup_name}")
    ax1.set_ylim(0, 1.2)

    # Pie chart for success vs failure probability
    labels = ['Success', 'Failure']
    sizes = [success_prob, failure_prob]
    colors = ['#66b3ff', '#ff9999']
    explode = (0.1, 0)  # explode the 1st slice (Success)

    ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
           autopct='%1.1f%%', shadow=True, startangle=90)
    ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax2.set_title(f"Prediction: {prediction_result}")

    plt.tight_layout()

    # Save visualization
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Prediction visualization saved to {save_path}")
    else:
        # If no save path provided, save to default location
        default_path = f"visualizations/{prediction['startup_name'].replace(' ', '_')}_prediction.png"
        if not os.path.exists(os.path.dirname(default_path)):
            os.makedirs(os.path.dirname(default_path))
        plt.savefig(default_path, dpi=300, bbox_inches='tight')
        print(f"✅ Prediction visualization saved to {default_path}")

    # Close the plot to avoid displaying it in non-interactive environments
    plt.close()


if __name__ == "__main__":
    # Example usage
    visualize_model_comparison()

    # Example prediction visualization
    example_prediction = {
        "startup_name": "TechStartup Inc",
        "success_probability": 75.5,
        "prediction": "Likely to IPO"
    }
    visualize_prediction(example_prediction)
