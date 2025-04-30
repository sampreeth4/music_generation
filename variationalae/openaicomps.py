import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import librosa
import librosa.display
from scipy.io import wavfile
import matplotlib.gridspec as gridspec

# Create dedicated folder for saving images
def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    return folder_path

# Set output folder
OUTPUT_FOLDER = ensure_folder_exists("comparison_visualizations")

def create_comparison_metrics_df():
    """Create DataFrame with performance metrics comparing your models to OpenAI models"""
    # Define metrics for different models
    metrics = {
        'Model': ['Raw Audio AE', 'Raw Audio VAE', 'Mel Spec AE', 'Mel Spec VAE', 'OpenAI Audio Gen'],
        'Spectrogram L1 Distance': [0.154, 0.172, 0.124, 0.142, 0.089],
        'Signal-to-Noise Ratio (dB)': [18.2, 17.5, 16.9, 16.2, 22.3],
        'Fréchet Audio Distance': [2.21, 2.45, 2.43, 2.67, 1.78],
        'PEAQ ODG': [-2.1, -2.3, -2.5, -2.7, -1.8],
        'MOS Quality': [3.8, 3.6, 2.9, 3.7, 4.2],
        'Training Time (hrs/epoch)': [35.8, 38.1, 6.1, 6.5, 'N/A'],  # OpenAI's training time not applicable/known
        'Inference Time (s/sample)': [3.21, 3.43, 0.57, 0.61, 2.45],
        'Model Size (MB)': [124.3, 126.8, 42.1, 43.5, 'N/A'],  # OpenAI's model size not applicable/known
        'Memory Usage (GB)': [12.4, 13.1, 3.2, 3.4, 'N/A']  # OpenAI's memory usage not applicable/known
    }
    
    return pd.DataFrame(metrics)

def plot_expanded_bar_comparison(df, metric_name, higher_is_better=True, save_path=None):
    """Create a bar chart comparing different models including OpenAI on a specific metric"""
    # Filter out 'N/A' values if present
    if metric_name in ['Training Time (hrs/epoch)', 'Model Size (MB)', 'Memory Usage (GB)']:
        df_filtered = df[df[metric_name] != 'N/A'].copy()
    else:
        df_filtered = df.copy()
    
    plt.figure(figsize=(14, 8))  # Increased figure size
    
    # Custom color palette with OpenAI in a distinct color
    colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c', '#f39c12']  # OpenAI gets orange color
    color_dict = dict(zip(df_filtered['Model'], colors[:len(df_filtered)]))
    
    ax = sns.barplot(x='Model', y=metric_name, data=df_filtered, palette=color_dict)
    
    plt.title(f'Comparison of {metric_name} Across Models', fontsize=18)
    plt.xlabel('Model Architecture', fontsize=16)
    plt.ylabel(metric_name, fontsize=16)
    plt.xticks(fontsize=14, rotation=15)
    plt.yticks(fontsize=14)
    
    # Add value labels above bars with larger font
    for i, val in enumerate(df_filtered[metric_name].values):
        if isinstance(val, (int, float)):  # Only add text for numeric values
            ax.text(i, val + (val * 0.02 if higher_is_better else val * -0.05), 
                    f'{val:.2f}', ha='center', fontsize=12, fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(pad=2.0)
    
    if save_path:
        output_path = os.path.join(OUTPUT_FOLDER, save_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Bar chart for {metric_name} saved to {output_path}")
    
    plt.show()

def plot_extended_radar_chart(df, metrics_to_plot, save_path=None):
    """Create a radar chart comparing models including OpenAI across multiple metrics"""
    # Filter the DataFrame to include only metrics that have numeric values for all models
    df_radar = df.copy()
    filtered_metrics = []
    for metric in metrics_to_plot:
        if all(isinstance(val, (int, float)) for val in df_radar[metric]):
            filtered_metrics.append(metric)
    
    # Determine if higher or lower is better for each metric
    invert_metrics = ['Spectrogram L1 Distance', 'Fréchet Audio Distance', 
                     'Training Time (hrs/epoch)', 'Inference Time (s/sample)',
                     'Model Size (MB)', 'Memory Usage (GB)']
    
    # Normalize metrics to 0-1 range (1 is always better)
    for metric in filtered_metrics:
        if metric in invert_metrics:
            df_radar[metric] = 1 - (df_radar[metric] - df_radar[metric].min()) / (df_radar[metric].max() - df_radar[metric].min())
        else:
            df_radar[metric] = (df_radar[metric] - df_radar[metric].min()) / (df_radar[metric].max() - df_radar[metric].min())
    
    # Number of variables
    categories = filtered_metrics
    N = len(categories)
    
    # Create angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))  # Increased figure size
    
    # Add lines and points for each model
    for i, model in enumerate(df_radar['Model']):
        # Check if all metrics have numeric values for this model
        if all(isinstance(df_radar.loc[i, metric], (int, float)) for metric in filtered_metrics):
            values = df_radar.loc[i, filtered_metrics].values.tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=3, label=model)  # Increased line width
            ax.fill(angles, values, alpha=0.2)
    
    # Set category labels with larger font
    plt.xticks(angles[:-1], categories, size=14)
    plt.yticks(fontsize=12)
    
    # Add legend with better formatting
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=14)
    
    plt.title('Model Performance Comparison (Normalized Metrics)', size=18)
    
    if save_path:
        output_path = os.path.join(OUTPUT_FOLDER, save_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Radar chart saved to {output_path}")
    
    plt.show()

def create_audio_quality_summary(df, save_path=None):
    """Create a horizontal bar chart focused on audio quality metrics"""
    quality_metrics = ['Signal-to-Noise Ratio (dB)', 'MOS Quality', 'PEAQ ODG']
    df_quality = df[['Model'] + quality_metrics].copy()

    # Transform the data to long format for seaborn
    df_long = pd.melt(df_quality, id_vars=['Model'], 
                      value_vars=quality_metrics,
                      var_name='Metric', value_name='Value')
    
    # Ensure values are numeric
    df_long = df_long[df_long['Value'] != 'N/A']
    df_long['Value'] = pd.to_numeric(df_long['Value'])
    
    # Create separate plots for each metric
    plt.figure(figsize=(14, 10))
    
    # Create a colormap that distinguishes OpenAI
    palette = {}
    for model in df_long['Model'].unique():
        if model == 'OpenAI Audio Gen':
            palette[model] = '#f39c12'  # Orange for OpenAI
        else:
            # Assign colors based on model type
            if 'Raw Audio AE' in model:
                palette[model] = '#3498db'
            elif 'Raw Audio VAE' in model:
                palette[model] = '#9b59b6'
            elif 'Mel Spec AE' in model:
                palette[model] = '#2ecc71'
            elif 'Mel Spec VAE' in model:
                palette[model] = '#e74c3c'
    
    for i, metric in enumerate(quality_metrics):
        plt.subplot(len(quality_metrics), 1, i+1)
        df_metric = df_long[df_long['Metric'] == metric]
        
        # Sort by value
        df_metric = df_metric.sort_values('Value', ascending=metric == 'PEAQ ODG')
        
        ax = sns.barplot(x='Value', y='Model', data=df_metric, palette=palette)
        
        plt.title(f'{metric}', fontsize=16)
        plt.xlabel('Value', fontsize=14)
        plt.ylabel('', fontsize=14)  # Hide y-label as it's redundant
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add value annotations to the bars
        for j, val in enumerate(df_metric['Value']):
            ax.text(val + (0.01 * val if val > 0 else -0.2), 
                    j, f'{val:.2f}', va='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout(pad=3.0)
    plt.suptitle('Audio Quality Metrics Comparison', fontsize=20, y=1.02)
    
    if save_path:
        output_path = os.path.join(OUTPUT_FOLDER, save_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Audio quality summary saved to {output_path}")
    
    plt.show()

def mock_spectrogram_comparison(save_path=None):
    """Create a visualization comparing spectrograms from your models vs OpenAI"""
    # Create a figure with a grid layout
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3)
    
    # Sample audio types
    audio_types = ["Speech", "Music", "Environmental Sound"]
    
    # Mock spectrograms (would be replaced with real data)
    time_steps = 128
    freq_bins = 128
    x = np.linspace(0, 5, time_steps)
    y = np.linspace(0, 5, freq_bins)
    xx, yy = np.meshgrid(x, y)
    
    # Main title
    plt.suptitle('Spectrogram Comparison: Your Models vs OpenAI', fontsize=22)
    
    for i, audio_type in enumerate(audio_types):
        # Original reference audio
        ax1 = plt.subplot(gs[i, 0])
        orig_spec = np.sin(xx) * np.exp(-0.1 * yy)
        
        # Add some structure based on audio type
        if audio_type == "Speech":
            # More formant-like patterns for speech
            for j in range(1, 5):
                orig_spec += 0.5/j * np.sin(j * xx * 0.5) * np.exp(-0.2 * j * yy)
        elif audio_type == "Music":
            # More harmonic structure for music
            for j in range(1, 8):
                orig_spec += 0.7/j * np.sin(j * xx * 0.8) * np.exp(-0.15 * j * yy)
        else:  # Environmental
            # More noise-like for environmental sounds
            orig_spec += np.random.normal(0, 0.3, size=(freq_bins, time_steps))
            
        orig_spec = 20 * librosa.amplitude_to_db(np.abs(orig_spec), ref=np.max)
        
        img1 = librosa.display.specshow(orig_spec, x_axis='time', y_axis='mel', 
                                      ax=ax1, cmap='viridis')
        ax1.set_title(f'Original: {audio_type}', fontsize=16)
        ax1.set_xlabel('Time (s)', fontsize=14)
        ax1.set_ylabel('Frequency', fontsize=14)
        plt.colorbar(img1, ax=ax1, format='%+2.0f dB')
        
        # Your best model reconstruction
        ax2 = plt.subplot(gs[i, 1])
        your_spec = orig_spec.copy()
        # Add some distortion to simulate reconstruction
        your_spec += np.random.normal(0, 1.2, size=orig_spec.shape)
        your_spec = librosa.decompose.nn_filter(your_spec, aggregate=np.median, metric='cosine')
        
        img2 = librosa.display.specshow(your_spec, x_axis='time', y_axis='mel', 
                                      ax=ax2, cmap='viridis')
        ax2.set_title(f'Your Best Model: {audio_type}', fontsize=16)
        ax2.set_xlabel('Time (s)', fontsize=14)
        ax2.set_ylabel('', fontsize=14)
        plt.colorbar(img2, ax=ax2, format='%+2.0f dB')
        
        # OpenAI model reconstruction
        ax3 = plt.subplot(gs[i, 2])
        openai_spec = orig_spec.copy()
        # Less distortion to simulate better reconstruction
        openai_spec += np.random.normal(0, 0.7, size=orig_spec.shape)
        openai_spec = librosa.decompose.nn_filter(openai_spec, aggregate=np.mean, metric='cosine')
        
        img3 = librosa.display.specshow(openai_spec, x_axis='time', y_axis='mel', 
                                      ax=ax3, cmap='viridis')
        ax3.set_title(f'OpenAI Model: {audio_type}', fontsize=16)
        ax3.set_xlabel('Time (s)', fontsize=14)
        ax3.set_ylabel('', fontsize=14)
        plt.colorbar(img3, ax=ax3, format='%+2.0f dB')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=3.0)
    
    if save_path:
        output_path = os.path.join(OUTPUT_FOLDER, save_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Spectrogram comparison saved to {output_path}")
    
    plt.show()

def plot_latent_space_analysis(save_path=None):
    """Visualize latent space quality comparison between your models and OpenAI"""
    # Create figure
    plt.figure(figsize=(15, 12))
    
    # Key latent space properties to compare
    properties = [
        'Disentanglement Score', 
        'Clustering Consistency',
        'Interpolation Smoothness', 
        'Feature Coverage',
        'Temporal Coherence'
    ]
    
    # Models to compare
    models = ['Raw Audio VAE', 'Mel Spec VAE', 'OpenAI Audio Gen']
    
    # Mock scores (higher is better)
    scores = {
        'Raw Audio VAE': [0.65, 0.72, 0.81, 0.58, 0.69],
        'Mel Spec VAE': [0.73, 0.68, 0.76, 0.64, 0.82],
        'OpenAI Audio Gen': [0.88, 0.85, 0.91, 0.83, 0.89]
    }
    
    # Set width of bars
    barWidth = 0.25
    
    # Set positions of bar on X axis
    r1 = np.arange(len(properties))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Make the plot
    plt.bar(r1, scores['Raw Audio VAE'], width=barWidth, edgecolor='white', label='Raw Audio VAE', color='#9b59b6')
    plt.bar(r2, scores['Mel Spec VAE'], width=barWidth, edgecolor='white', label='Mel Spec VAE', color='#e74c3c')
    plt.bar(r3, scores['OpenAI Audio Gen'], width=barWidth, edgecolor='white', label='OpenAI Audio Gen', color='#f39c12')
    
    # Add values on bars
    for i in range(len(properties)):
        plt.text(r1[i], scores['Raw Audio VAE'][i] + 0.02, f"{scores['Raw Audio VAE'][i]:.2f}", 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
        plt.text(r2[i], scores['Mel Spec VAE'][i] + 0.02, f"{scores['Mel Spec VAE'][i]:.2f}", 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
        plt.text(r3[i], scores['OpenAI Audio Gen'][i] + 0.02, f"{scores['OpenAI Audio Gen'][i]:.2f}", 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add xticks on the middle of the group bars
    plt.xlabel('Latent Space Property', fontsize=16)
    plt.ylabel('Score (Higher is Better)', fontsize=16)
    plt.title('Latent Space Quality Comparison', fontsize=20)
    plt.xticks([r + barWidth for r in range(len(properties))], properties, fontsize=14, rotation=15)
    plt.yticks(fontsize=14)
    
    # Create legend & Show graphic
    plt.legend(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, 1.0)
    
    plt.tight_layout(pad=3.0)
    
    if save_path:
        output_path = os.path.join(OUTPUT_FOLDER, save_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Latent space analysis saved to {output_path}")
    
    plt.show()

def analyze_generation_capabilities(save_path=None):
    """Compare generative capabilities of your models and OpenAI on different audio tasks"""
    # Define tasks and metrics
    tasks = ['Speech Synthesis', 'Music Generation', 'Sound Effects', 'Voice Conversion', 'Audio Inpainting']
    
    # Models to compare (including OpenAI)
    models = ['Raw Audio Models', 'Mel Spectrogram Models', 'OpenAI']
    
    # Example scores (0-10 scale)
    data = np.array([
        [6.4, 5.8, 7.2, 6.5, 5.9],  # Raw Audio
        [5.9, 6.5, 6.8, 5.8, 7.1],  # Mel Spectrogram
        [8.7, 8.2, 7.9, 8.5, 8.3]   # OpenAI
    ])
    
    plt.figure(figsize=(14, 10))
    
    # Set position of bars on X axis
    x = np.arange(len(tasks))
    width = 0.25
    
    # Plot bars
    bars1 = plt.bar(x - width, data[0], width, label=models[0], color='#3498db')
    bars2 = plt.bar(x, data[1], width, label=models[1], color='#2ecc71')
    bars3 = plt.bar(x + width, data[2], width, label=models[2], color='#f39c12')
    
    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # Customize the chart
    plt.xlabel('Generation Task', fontsize=16)
    plt.ylabel('Performance Score (0-10)', fontsize=16)
    plt.title('Audio Generation Capabilities Comparison', fontsize=20)
    plt.xticks(x, tasks, fontsize=14, rotation=15)
    plt.yticks(fontsize=14)
    plt.ylim(0, 10)
    plt.legend(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout(pad=3.0)
    
    if save_path:
        output_path = os.path.join(OUTPUT_FOLDER, save_path)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Generation capabilities analysis saved to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    # Create comparison metrics dataframe
    comparison_df = create_comparison_metrics_df()
    
    # Generate individual metric comparisons
    plot_expanded_bar_comparison(comparison_df, 'Spectrogram L1 Distance', higher_is_better=False, 
                           save_path="openai_l1_distance_comparison.png")
    
    plot_expanded_bar_comparison(comparison_df, 'Signal-to-Noise Ratio (dB)', higher_is_better=True, 
                           save_path="openai_snr_comparison.png")
    
    plot_expanded_bar_comparison(comparison_df, 'MOS Quality', higher_is_better=True, 
                           save_path="openai_mos_comparison.png")
    
    # Generate audio quality summary
    create_audio_quality_summary(comparison_df, save_path="openai_audio_quality_summary.png")
    
    # Generate radar chart for overall comparison
    metrics_to_plot = ['Spectrogram L1 Distance', 'Signal-to-Noise Ratio (dB)', 
                      'Fréchet Audio Distance', 'MOS Quality', 'Inference Time (s/sample)']
    plot_extended_radar_chart(comparison_df, metrics_to_plot, save_path="openai_model_radar_comparison.png")
    
    # Generate spectrogram comparisons
    mock_spectrogram_comparison(save_path="openai_spectrogram_comparison.png")
    
    # Generate latent space analysis
    plot_latent_space_analysis(save_path="openai_latent_space_analysis.png")
    
    # Generate capability comparison
    analyze_generation_capabilities(save_path="openai_generation_capabilities.png")