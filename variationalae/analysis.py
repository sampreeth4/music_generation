import numpy as np
import matplotlib.pyplot as plt
import os
from vae import VAE
from train import load_mnist

# Create dedicated folder for saving images
def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")
    return folder_path

# Set output folder
OUTPUT_FOLDER = ensure_folder_exists("visualizations")

def select_images(images, labels, num_images=10):
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels

def plot_reconstructed_images(images, reconstructed_images):
    fig = plt.figure(figsize=(18, 5))  # Increased figure size
    num_images = len(images)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.2)  # Adjust spacing
    
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        
        # Add row label for the first column
        if i == 0:
            ax.set_title("Original", fontsize=16, y=-0.2)
        
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
        
        # Add row label for the first column
        if i == 0:
            ax.set_title("Reconstructed", fontsize=16, y=-0.2)
    
    plt.tight_layout(pad=2.0)  # Increase padding around subplots
    
    # Save figure in the output folder
    output_path = os.path.join(OUTPUT_FOLDER, 'reconstructed_images.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved reconstructed images to: {output_path}")
    
    plt.show()

def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    plt.figure(figsize=(12, 10))  # Increased figure size
    
    scatter = plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.7,  # Increased alpha for better visibility
                s=15)       # Increased marker size
    
    # Add title and labels with larger font
    plt.title('Latent Space Representation', fontsize=18)
    plt.xlabel('Latent Dimension 1', fontsize=16)
    plt.ylabel('Latent Dimension 2', fontsize=16)
    
    # Improve tick label size
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Add colorbar with improved formatting
    cbar = plt.colorbar(scatter)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Digit Class', fontsize=16)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Save figure in the output folder
    output_path = os.path.join(OUTPUT_FOLDER, 'latent_space.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved latent space visualization to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    autoencoder = VAE.load("model")
    x_train, y_train, x_test, y_test = load_mnist()
    
    # Plot reconstructed images
    num_sample_images_to_show = 8
    sample_images, _ = select_images(x_test, y_test, num_sample_images_to_show)
    reconstructed_images = autoencoder.reconstruct(sample_images)
    plot_reconstructed_images(sample_images, reconstructed_images)
    
    # Plot latent space
    num_images = 6000
    sample_images, sample_labels = select_images(x_test, y_test, num_images)
    latent_representations = autoencoder.encoder.predict(sample_images)[2]  # that's z
    plot_images_encoded_in_latent_space(latent_representations, sample_labels)