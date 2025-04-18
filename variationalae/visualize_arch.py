# File: visualize_architecture.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, Lambda
import matplotlib.gridspec as gridspec

def plot_layer_box(ax, x, y, width, height, layer_name, color='lightblue', alpha=0.8, fontsize=8):
    """Helper function to draw a box representing a layer"""
    rect = Rectangle((x, y), width, height, facecolor=color, alpha=alpha, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, layer_name, ha='center', va='center', fontsize=fontsize)

def draw_arrow(ax, x1, y1, x2, y2, color='black'):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', 
                          connectionstyle='arc3,rad=0.0', linewidth=1.5, color=color)
    ax.add_patch(arrow)

def visualize_architecture(is_vae=True, save_path=None):
    """Visualize the architecture of AE or VAE"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up the coordinate system
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # Draw the architecture components
    # Input
    plot_layer_box(ax, 0.5, 2, 1, 1, 'Input\n(audio/spec)', color='lightyellow')
    
    # Encoder layers
    plot_layer_box(ax, 2, 2, 1, 1, 'Conv2D\n32 filters\n3x3, stride 1', color='lightblue')
    plot_layer_box(ax, 3, 2, 1, 1, 'Conv2D\n64 filters\n3x3, stride 2', color='lightblue')
    plot_layer_box(ax, 4, 2, 1, 1, 'Conv2D\n64 filters\n3x3, stride 2', color='lightblue')
    plot_layer_box(ax, 5, 2, 1, 1, 'Conv2D\n64 filters\n3x3, stride 1', color='lightblue')
    plot_layer_box(ax, 6, 2, 1, 1, 'Flatten', color='lightgreen')
    
    # Latent space
    if is_vae:
        plot_layer_box(ax, 7, 2.5, 1, 0.5, 'μ', color='lightsalmon')
        plot_layer_box(ax, 7, 1.5, 1, 0.5, 'log σ²', color='lightsalmon')
        plot_layer_box(ax, 8, 2, 1, 1, 'Sampling\nz = μ + σε', color='lightpink')
        title = 'Variational Autoencoder (VAE) Architecture'
    else:
        plot_layer_box(ax, 7, 2, 1, 1, 'Dense\n(latent space)', color='lightsalmon')
        title = 'Autoencoder (AE) Architecture'
    
    # Decoder layers  
    plot_layer_box(ax, 9, 2, 1, 1, 'Reconstruction', color='lightyellow')
    
    # Decoder details below
    y_decoder = 0.5
    plot_layer_box(ax, 2, y_decoder, 1, 0.5, 'Dense\nReshape', color='lightgreen')
    plot_layer_box(ax, 3, y_decoder, 1, 0.5, 'ConvT2D\n64 filters', color='lightblue')
    plot_layer_box(ax, 4, y_decoder, 1, 0.5, 'ConvT2D\n64 filters', color='lightblue')
    plot_layer_box(ax, 5, y_decoder, 1, 0.5, 'ConvT2D\n32 filters', color='lightblue')
    plot_layer_box(ax, 6, y_decoder, 1, 0.5, 'ConvT2D\n1 filter', color='lightblue')
    
    # Draw the arrows for encoder
    draw_arrow(ax, 1.5, 2.5, 2, 2.5)
    draw_arrow(ax, 3, 2.5, 3.5, 2.5)
    draw_arrow(ax, 4, 2.5, 4.5, 2.5)
    draw_arrow(ax, 5, 2.5, 5.5, 2.5)
    draw_arrow(ax, 6, 2.5, 6.5, 2.5)
    
    if is_vae:
        draw_arrow(ax, 7, 3, 7.5, 2.5)
        draw_arrow(ax, 7, 1.5, 7.5, 2)
        draw_arrow(ax, 8, 2.5, 9, 2.5)
    else:
        draw_arrow(ax, 7, 2.5, 9, 2.5)
    
    # Additional arrows for decoder flow
    if is_vae:
        draw_arrow(ax, 8, 1.5, 2, 0.75)
    else:
        draw_arrow(ax, 7, 1.5, 2, 0.75)
        
    draw_arrow(ax, 3, 0.75, 3.5, 0.75)
    draw_arrow(ax, 4, 0.75, 4.5, 0.75)
    draw_arrow(ax, 5, 0.75, 5.5, 0.75)
    draw_arrow(ax, 6, 0.75, 9, 1.75)
    
    plt.title(title, fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Architecture diagram saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Generate both architecture diagrams
    visualize_architecture(is_vae=True, save_path="vae_architecture.png")
    visualize_architecture(is_vae=False, save_path="ae_architecture.png")