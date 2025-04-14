import os
import pickle

import numpy as np
import soundfile as sf

from soundgenerator import SoundGenerator
from autoencoder import VAE
from train import SPECTROGRAM_PATH


HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = "C:/Users/A.SREE SAI SAMPREETH/OneDrive/Documents/GitHub/music_generation/generation/samples/original/"
SAVE_DIR_GENERATED = "C:/Users/A.SREE SAI SAMPREETH/OneDrive/Documents/GitHub/music_generation/generation/samples/generated/"
MIN_MAX_VALUES_PATH = "C:/Users/A.SREE SAI SAMPREETH/OneDrive/Documents/GitHub/music_generation/generation/min_max_values.pkl"
# SPECTROGRAM_PATH = "C:/Users/A.SREE SAI SAMPREETH/OneDrive/Documents/GitHub/music_generation/generation/spectrograms/"

def load_fsdd(spectrograms_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) # (n_bins, n_frames, 1)
            x_train.append(spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] # -> (3000, 256, 64, 1)
    return x_train, file_paths


def select_spectrograms(spectrograms,
                        file_paths,
                        min_max_values,
                        num_spectrograms=2):
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrogrmas = spectrograms[sampled_indexes]
    sampled_file_paths = [file_paths[index] for index in sampled_indexes]
    
    print("Selected files:", sampled_file_paths)
    
    # Create default min_max_values if not found
    sampled_min_max_values = []
    
    # Check if we have any valid keys in min_max_values
    valid_keys = [k for k in min_max_values.keys() if k is not None]
    
    if valid_keys:
        # If we have valid keys, use the first one's values as a template
        template_values = min_max_values[valid_keys[0]]
        default_min = template_values["min"]
        default_max = template_values["max"]
    else:
        # If no valid keys, use reasonable defaults for audio spectrograms
        default_min = -100.0  # A common minimum dB value
        default_max = 0.0     # A common maximum dB value
    
    print(f"Using default min: {default_min}, max: {default_max}")
    
    # Create min_max values for each sample
    for _ in sampled_file_paths:
        sampled_min_max_values.append({
            "min": default_min,
            "max": default_max
        })
    
    return sampled_spectrogrmas, sampled_min_max_values


def save_signals(signals, save_dir, sample_rate=22050):
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)


if __name__ == "__main__":
    # initialise sound generator
    vae = VAE.load("model")
    sound_generator = SoundGenerator(vae, HOP_LENGTH)

    # load spectrograms + min max values
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)

    specs, file_paths = load_fsdd(SPECTROGRAM_PATH)

    # sample spectrograms + min max values
    sampled_specs, sampled_min_max_values = select_spectrograms(specs,
                                                                file_paths,
                                                                min_max_values,
                                                                5)

    # generate audio for sampled spectrograms
    signals, _ = sound_generator.generate(sampled_specs,
                                          sampled_min_max_values)

    # convert spectrogram samples to audio
    original_signals = sound_generator.convert_spectrograms_to_audio(
        sampled_specs, sampled_min_max_values)

    # save audio signals
    save_signals(signals, SAVE_DIR_GENERATED)
    save_signals(original_signals, SAVE_DIR_ORIGINAL)





