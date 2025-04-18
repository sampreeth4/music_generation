# File: visualize_metrics.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def create_metrics_df():
    """Create DataFrame with performance metrics for different model configurations"""
    # Define metrics for different models
    metrics = {
        'Model': ['Raw Audio AE', 'Raw Audio VAE', 'Mel Spec AE', 'Mel Spec VAE'],
        'Spectrogram L1 Distance': [0.154, 0.172, 0.124, 0.142],
        'Signal-to-Noise Ratio (dB)': [18.2, 17.5, 16.9, 16.2],
        'Fréchet Audio Distance': [2.21, 2.45, 2.43, 2.67],
        'PEAQ ODG': [-2.1, -2.3, -2.5, -2.7],
        'MOS Quality': [3.8, 3.6, 2.9, 3.7],
        'Training Time (hrs/epoch)': [35.8, 38.1, 6.1, 6.5],
        'Inference Time (s/sample)': [3.21, 3.43, 0.57, 0.61],
        'Model Size (MB)': [124.3, 126.8, 42.1, 43.5],
        'Memory Usage (GB)': [12.4, 13.1, 3.2, 3.4]
    }
    
    return pd.DataFrame(metrics)

def plot_bar_comparison(df, metric_name, higher_is_better=True, save_path=None):
    """Create a bar chart comparing different models on a specific metric"""
    plt.figure(figsize=(10, 6))
    
    colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c']
    color_dict = dict(zip(df['Model'], colors))
    
    ax = sns.barplot(x='Model', y=metric_name, data=df, palette=color_dict)
    
    plt.title(f'Comparison of {metric_name} Across Models', fontsize=14)
    plt.xlabel('Model Architecture', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.xticks(rotation=15)
    
    # Add value labels above bars
    for i, val in enumerate(df[metric_name].values):
        ax.text(i, val + (val * 0.02 if higher_is_better else val * -0.05), 
                f'{val:.2f}', ha='center', fontsize=10)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bar chart for {metric_name} saved to {save_path}")
    
    plt.show()

def plot_radar_chart(df, metrics_to_plot, save_path=None):
    """Create a radar chart comparing models across multiple metrics"""
    # Normalize metrics for radar chart
    df_radar = df.copy()
    
    # Determine if higher or lower is better for each metric
    invert_metrics = ['Spectrogram L1 Distance', 'Fréchet Audio Distance', 
                     'Training Time (hrs/epoch)', 'Inference Time (s/sample)',
                     'Model Size (MB)', 'Memory Usage (GB)']
    
    # Normalize metrics to 0-1 range (1 is always better)
    for metric in metrics_to_plot:
        if metric in invert_metrics:
            df_radar[metric] = 1 - (df_radar[metric] - df_radar[metric].min()) / (df_radar[metric].max() - df_radar[metric].min())
        else:
            df_radar[metric] = (df_radar[metric] - df_radar[metric].min()) / (df_radar[metric].max() - df_radar[metric].min())
    
    # Number of variables
    categories = metrics_to_plot
    N = len(categories)
    
    # Create angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add lines and points for each model
    for i, model in enumerate(df_radar['Model']):
        values = df_radar.loc[i, metrics_to_plot].values.tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Set category labels
    plt.xticks(angles[:-1], categories, size=10)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Model Performance Comparison (Normalized Metrics)', size=15)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Radar chart saved to {save_path}")
    
    plt.show()

def plot_training_curves(save_path=None):
    """Create training loss curves for AE and VAE models"""
    # Mock training history data
    epochs = np.arange(1, 101)
    
    # Create loss curves with realistic decay patterns
    ae_raw_loss = 0.5 * np.exp(-0.03 * epochs) + 0.15 + np.random.normal(0, 0.01, size=len(epochs))
    vae_raw_loss = 0.6 * np.exp(-0.025 * epochs) + 0.2 + np.random.normal(0, 0.015, size=len(epochs))
    ae_spec_loss = 0.45 * np.exp(-0.04 * epochs) + 0.12 + np.random.normal(0, 0.008, size=len(epochs))
    vae_spec_loss = 0.55 * np.exp(-0.035 * epochs) + 0.17 + np.random.normal(0, 0.012, size=len(epochs))
    
    # For VAE, add KL divergence loss component
    vae_raw_kl = 0.2 * (1 - np.exp(-0.05 * epochs)) + np.random.normal(0, 0.005, size=len(epochs))
    vae_spec_kl = 0.15 * (1 - np.exp(-0.06 * epochs)) + np.random.normal(0, 0.004, size=len(epochs))
    
    plt.figure(figsize=(12, 8))
    
    # Plot reconstruction loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, ae_raw_loss, label='AE - Raw Audio', color='#3498db')
    plt.plot(epochs, vae_raw_loss, label='VAE - Raw Audio', color='#9b59b6')
    plt.plot(epochs, ae_spec_loss, label='AE - Mel Spectrogram', color='#2ecc71')
    plt.plot(epochs, vae_spec_loss, label='VAE - Mel Spectrogram', color='#e74c3c')
    
    plt.title('Training Loss Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Reconstruction Loss', fontsize=12)
    plt.grid(linestyle='--', alpha=0.6)
    plt.legend()
    
    # Plot KL divergence loss for VAEs
    plt.subplot(2, 1, 2)
    plt.plot(epochs, vae_raw_kl, label='VAE - Raw Audio', color='#9b59b6')
    plt.plot(epochs, vae_spec_kl, label='VAE - Mel Spectrogram', color='#e74c3c')
    
    plt.title('KL Divergence Loss for VAE Models', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('KL Divergence Loss', fontsize=12)
    plt.grid(linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()

def plot_computational_efficiency(df, save_path=None):
    """Create a visualization comparing computational efficiency metrics"""
    plt.figure(figsize=(14, 10))
    
    # Prepare data for grouped bar chart
    models = df['Model'].tolist()
    training_time = df['Training Time (hrs/epoch)'].tolist()
    inference_time = df['Inference Time (s/sample)'].tolist()
    model_size = df['Model Size (MB)'].tolist()
    memory_usage = df['Memory Usage (GB)'].tolist()
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training Time
    bars1 = ax[0, 0].bar(models, training_time, color=['#3498db', '#9b59b6', '#2ecc71', '#e74c3c'])
    ax[0, 0].set_title('Training Time (hrs/epoch)')
    ax[0, 0].set_ylabel('Hours')
    ax[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Inference Time
    bars2 = ax[0, 1].bar(models, inference_time, color=['#3498db', '#9b59b6', '#2ecc71', '#e74c3c'])
    ax[0, 1].set_title('Inference Time (s/sample)')
    ax[0, 1].set_ylabel('Seconds')
    ax[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Model Size
    bars3 = ax[1, 0].bar(models, model_size, color=['#3498db', '#9b59b6', '#2ecc71', '#e74c3c'])
    ax[1, 0].set_title('Model Size (MB)')
    ax[1, 0].set_ylabel('Megabytes')
    ax[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Memory Usage
    bars4 = ax[1, 1].bar(models, memory_usage, color=['#3498db', '#9b59b6', '#2ecc71', '#e74c3c'])
    ax[1, 1].set_title('Memory Usage (GB)')
    ax[1, 1].set_ylabel('Gigabytes')
    ax[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on all bars
    for i, ax_i in enumerate(ax.flatten()):
        bars = [bars1, bars2, bars3, bars4][i]
        for bar in bars:
            height = bar.get_height()
            ax_i.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            
    plt.suptitle('Computational Efficiency Comparison', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Computational efficiency visualization saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Create the metrics dataframe
    metrics_df = create_metrics_df()
    
    # Generate individual metric comparisons
    plot_bar_comparison(metrics_df, 'Spectrogram L1 Distance', higher_is_better=False, 
                       save_path="l1_distance_comparison.png")
    
    plot_bar_comparison(metrics_df, 'Signal-to-Noise Ratio (dB)', higher_is_better=True, 
                       save_path="snr_comparison.png")
    
    plot_bar_comparison(metrics_df, 'MOS Quality', higher_is_better=True, 
                       save_path="mos_comparison.png")
    
    # Generate radar chart for overall comparison
    metrics_to_plot = ['Spectrogram L1 Distance', 'Signal-to-Noise Ratio (dB)', 
                      'Fréchet Audio Distance', 'MOS Quality', 'Inference Time (s/sample)']
    plot_radar_chart(metrics_df, metrics_to_plot, save_path="model_radar_comparison.png")
    
    # Generate training curves
    plot_training_curves(save_path="training_curves.png")
    
    # Generate computational efficiency visualization
    plot_computational_efficiency(metrics_df, save_path="computational_efficiency.png")