import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import pickle
from scipy.io import wavfile
import matplotlib.gridspec as gridspec

def load_audio_sample(file_path, duration=4.0, sr=44100):
    """Load audio file and return as numpy array"""
    audio, sr = librosa.load(file_path, sr=sr, duration=duration)
    return audio, sr

def extract_mel_spectrogram(audio, sr, n_fft=1024, hop_length=256, n_mels=128):
    """Extract mel spectrogram from audio"""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, 
                                            hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def reconstruct_spectrogram(model, spectrogram, is_vae=True):
    """Reconstruct spectrogram using the trained model"""
    # Add batch dimension and channel dimension if needed
    input_spec = spectrogram.T
    if len(input_spec.shape) == 2:
        input_spec = np.expand_dims(input_spec, axis=0)
        if input_spec.shape[-1] != 1:
            input_spec = np.expand_dims(input_spec, axis=-1)
    
    # Model inference
    if is_vae:
        _, _, z = model.encoder.predict(input_spec)
        reconstructed = model.decoder.predict(z)[0]
    else:
        reconstructed = model.predict(input_spec)[0]
    
    # Remove extra dimension if present
    if len(reconstructed.shape) > 2 and reconstructed.shape[-1] == 1:
        reconstructed = reconstructed.squeeze(-1)
    
    return reconstructed

def visualize_spectrograms(original_specs, reconstructed_specs, audio_names, is_vae=True, save_path=None):
    """Visualize original and reconstructed spectrograms side by side"""
    n_samples = len(original_specs)
    fig = plt.figure(figsize=(15, 3 * n_samples))
    
    model_type = "VAE" if is_vae else "AE"
    plt.suptitle(f'Original vs {model_type} Reconstructed Spectrograms', fontsize=16)
    
    for i in range(n_samples):
        # Original spectrogram
        plt.subplot(n_samples, 2, 2*i + 1)
        librosa.display.specshow(original_specs[i], x_axis='time', y_axis='mel', sr=44100, 
                                hop_length=256, cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Original: {audio_names[i]}')
        
        # Reconstructed spectrogram
        plt.subplot(n_samples, 2, 2*i + 2)
        librosa.display.specshow(reconstructed_specs[i], x_axis='time', y_axis='mel', sr=44100,
                                hop_length=256, cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Reconstructed ({model_type})')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spectrogram comparison saved to {save_path}")
    
    plt.show()

def compare_artifact_assessment(artifact_scores, save_path=None):
    """Create visualization for artifact assessment scores"""
    categories = ['Temporal\nDiscontinuities', 'Frequency\nSmearing', 'Phase\nInconsistencies', 
                 'Harmonic\nDistortion', 'Transient\nPreservation']
    
    ae_scores = [3.2, 2.7, 3.5, 2.9, 2.2]  # Example scores for AE
    vae_scores = [2.5, 2.3, 2.8, 2.1, 3.5]  # Example scores for VAE
    spec_raw_scores = [2.8, 1.9, 4.2, 2.5, 2.7]  # Example scores for Raw Audio
    
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, ae_scores, width, label='AE', color='lightblue')
    bars2 = ax.bar(x, vae_scores, width, label='VAE', color='salmon')
    bars3 = ax.bar(x + width, spec_raw_scores, width, label='Raw Audio', color='lightgreen')
    
    # Add some text for labels and custom x-axis tick labels, etc.
    ax.set_ylabel('Artifact Severity Score (0-5)', fontsize=12)
    ax.set_title('Audio Artifact Assessment by Model Type', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 5)
    
    # Add value labels on top of bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Artifact assessment visualization saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Sample code to demonstrate usage
    # In a real scenario, you would load your trained models and audio samples
    
    # Mock spectrograms for demonstration
    sample_names = ["Piano", "Violin", "Voice", "Drums"]
    orig_specs = []
    reconstructed_specs_ae = []
    reconstructed_specs_vae = []
    
    # Create mock data for visualization
    for i in range(4):
        # Create random spectrograms with structure to simulate real data
        time_steps = 128
        freq_bins = 128
        
        # Original spectrogram with some structure
        x = np.linspace(0, 5, time_steps)
        y = np.linspace(0, 5, freq_bins)
        xx, yy = np.meshgrid(x, y)
        orig_spec = np.sin(xx) * np.exp(-0.1 * yy) 
        orig_spec += np.random.normal(0, 0.05, size=(freq_bins, time_steps))
        
        # Add some harmonic structure
        for j in range(1, 5):
            orig_spec += 0.5/j * np.sin(j * xx) * np.exp(-0.2 * j * yy)
        
        # Scale to dB range
        orig_spec = 20 * librosa.amplitude_to_db(np.abs(orig_spec), ref=np.max)
        
        # Reconstructed with AE (slightly blurrier)
        recon_ae = orig_spec.copy() 
        recon_ae += np.random.normal(0, 1.0, size=orig_spec.shape)
        recon_ae = librosa.decompose.nn_filter(recon_ae, aggregate=np.median, metric='cosine')
        
        # Reconstructed with VAE (smoother)
        recon_vae = orig_spec.copy()
        recon_vae += np.random.normal(0, 0.8, size=orig_spec.shape)
        recon_vae = librosa.decompose.nn_filter(recon_vae, aggregate=np.mean, metric='cosine')
        
        orig_specs.append(orig_spec)
        reconstructed_specs_ae.append(recon_ae)
        reconstructed_specs_vae.append(recon_vae)
    
    # Create visualizations
    visualize_spectrograms(orig_specs[:2], reconstructed_specs_ae[:2], sample_names[:2], 
                          is_vae=False, save_path="ae_spectrogram_comparison.png")
    
    visualize_spectrograms(orig_specs[2:], reconstructed_specs_vae[2:], sample_names[2:], 
                          is_vae=True, save_path="vae_spectrogram_comparison.png")
    
    # Artifact assessment visualization
    compare_artifact_assessment(None, save_path="artifact_assessment.png")