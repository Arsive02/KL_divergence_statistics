import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image
import io
import seaborn as sns
import torchvision
import os
import imageio

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=2):
        super(VAE, self).__init__()
        
        # Encoder network: compresses input into latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # These layers output parameters of the latent distribution
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)    # Mean of latent distribution
        self.fc_var = nn.Linear(hidden_dim, latent_dim)   # Log variance of latent distribution
        
        # Decoder network: reconstructs input from latent representation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Ensures output is between 0 and 1 (like MNIST pixels)
        )
        
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, logvar):
        # Implementation of the reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Random noise from standard normal
        return mu + eps * std        # Transformed noise gives us our latent sample
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar, beta=1.0):
    """
    Computes the VAE loss function with separate reconstruction and KL terms.
    beta parameter controls the trade-off between reconstruction and regularization.
    """
    # Reconstruction loss (how well we can reconstruct the input)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL divergence loss (how close our latent distribution is to standard normal)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + beta * KLD, BCE, KLD

def create_training_animation():
    """
    Creates an animation of the training process of the VAE.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    
    # Initialize model and optimizer
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    frames = []
    n_epochs = 50
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 5))
    gs = plt.GridSpec(1, 4)
    ax1 = fig.add_subplot(gs[0, 0])  # Latent space distributions
    ax2 = fig.add_subplot(gs[0, 1])  # Sample reconstructions
    ax3 = fig.add_subplot(gs[0, 2])  # KL divergence over time
    ax4 = fig.add_subplot(gs[0, 3])  # Reconstruction loss over time
    
    # Training history
    kl_history = []
    bce_history = []
    
    # Generate samples from true prior (standard normal)
    true_samples = np.random.normal(0, 1, (1000, 2))
    
    for epoch in tqdm(range(n_epochs)):
        model.train()
        epoch_kl = 0
        epoch_bce = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = model(data)
            loss, bce, kld = loss_function(recon_batch, data, mu, logvar, beta=1.0)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_kl += kld.item()
            epoch_bce += bce.item()
        
        # Calculate average losses for the epoch
        avg_kl = epoch_kl / len(train_loader.dataset)
        avg_bce = epoch_bce / len(train_loader.dataset)
        kl_history.append(avg_kl)
        bce_history.append(avg_bce)
        
        # Visualization
        model.eval()
        with torch.no_grad():
            test_data, _ = next(iter(train_loader))
            test_data = test_data.to(device)
            recon, mu, logvar = model(test_data)
            
            # Clear previous plots
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            
            # Plot 1: Distribution comparison
            latent_samples = mu.cpu().numpy()
            sns.kdeplot(data=true_samples[:, 0], ax=ax1, color='blue', label='Prior', alpha=0.5)
            sns.kdeplot(data=latent_samples[:, 0], ax=ax1, color='red', label='Encoded', alpha=0.5)
            ax1.set_title(f'Latent Space Distribution\n(Epoch {epoch+1})')
            ax1.legend()
            
            # Plot 2: Original vs Reconstructed
            n = 8
            comparison = torch.cat([test_data[:n], recon.view(128, 1, 28, 28)[:n]])
            img_grid = torchvision.utils.make_grid(comparison, nrow=n)
            ax2.imshow(img_grid.cpu().numpy().transpose((1, 2, 0)), cmap='gray')
            ax2.set_title('Original (top) vs\nReconstructed (bottom)')
            ax2.axis('off')
            
            # Plot 3: KL Divergence over time
            ax3.plot(range(1, epoch + 2), kl_history, 'g-')
            ax3.set_title('KL Divergence Over Time')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('KL Divergence')
            ax3.grid(True)
            
            # Plot 4: Reconstruction Loss over time
            ax4.plot(range(1, epoch + 2), bce_history, 'b-')
            ax4.set_title('Reconstruction Loss Over Time')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('BCE Loss')
            ax4.grid(True)
            
            plt.tight_layout()
            
            # Save the plot as a frame
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            img = Image.open(buf)
            frames.append(img)
    
    # Save animation as GIF
    frames[0].save(
        'vae_training_detailed.gif',
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=200,
        loop=0
    )
    
    print("Animation saved as vae_training_detailed.gif")
    
    # Plot final loss curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(kl_history, label='KL Divergence')
    plt.title('KL Divergence Over Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(bce_history, label='Reconstruction Loss')
    plt.title('Reconstruction Loss Over Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('final_loss_curves.png')
    plt.close()

def visualize_latent_space(model, device, test_loader, epoch, save_path=None):
    """
    Creates three visualizations of the latent space:
    1. Scatter plot of encoded points colored by digit class
    2. Density plot showing the overall distribution
    3. Grid of decoded images from different points in latent space
    """
    model.eval()
    
    # Create a large figure with three subplots
    fig = plt.figure(figsize=(20, 6))
    
    # Plot 1: Scatter plot of encoded points
    ax1 = fig.add_subplot(131)
    
    # Store latent vectors and corresponding labels
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            mu, logvar = model.encode(data.view(-1, 784))
            z = model.reparameterize(mu, logvar)
            latent_vectors.append(z.cpu().numpy())
            labels.extend(target.numpy())
    
    # Concatenate all latent vectors
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.array(labels)
    
    # Create scatter plot
    scatter = ax1.scatter(latent_vectors[:, 0], latent_vectors[:, 1], 
                         c=labels, cmap='tab10', alpha=0.6, s=10)
    ax1.set_title(f'Latent Space Representation\nEpoch {epoch}')
    ax1.set_xlabel('First Latent Dimension')
    ax1.set_ylabel('Second Latent Dimension')
    plt.colorbar(scatter, ax=ax1, label='Digit Class')
    
    # Plot 2: Density plot
    ax2 = fig.add_subplot(132)
    sns.kdeplot(data=latent_vectors, ax=ax2, fill=True)
    ax2.set_title('Latent Space Density')
    ax2.set_xlabel('First Latent Dimension')
    ax2.set_ylabel('Second Latent Dimension')
    
    # Plot 3: Generate images from latent space grid
    ax3 = fig.add_subplot(133)
    
    # Create a grid of points in latent space
    x = np.linspace(-3, 3, 20)
    y = np.linspace(-3, 3, 20)
    xx, yy = np.meshgrid(x, y)
    
    # Sample points from grid
    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    
    # Generate images from grid points
    with torch.no_grad():
        z = torch.FloatTensor(grid_points).to(device)
        decoded = model.decode(z)
        
    # Create image grid
    n = 20
    img_size = 28
    canvas = np.zeros((n * img_size, n * img_size))
    
    for i in range(n):
        for j in range(n):
            idx = i * n + j
            if idx < decoded.shape[0]:
                canvas[i * img_size:(i + 1) * img_size, 
                       j * img_size:(j + 1) * img_size] = \
                    decoded[idx].cpu().view(img_size, img_size)
    
    ax3.imshow(canvas, cmap='gray')
    ax3.set_title('Latent Space Traversal')
    ax3.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

def create_latent_space_animation(n_epochs=50):
    """
    Creates an animation of the latent space evolution during training.
    """
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    
    # Initialize model
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Create directory for frames
    os.makedirs('latent_viz', exist_ok=True)
    
    # Training loop
    for epoch in tqdm(range(n_epochs)):
        # Training code here (same as before)
        model.train()
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, _, _ = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            optimizer.step()
        
        # Visualize latent space
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            visualize_latent_space(model, device, test_loader, epoch + 1,
                                 f'latent_viz/latent_space_epoch_{epoch+1:03d}.png')
    
    # Create GIF from saved frames
    images = []
    for epoch in range(0, n_epochs, 5):
        filename = f'latent_viz/latent_space_epoch_{epoch+1:03d}.png'
        images.append(imageio.imread(filename))
    
    imageio.mimsave('latent_space_evolution.gif', images, duration=1.0)
    print("Animation saved as latent_space_evolution.gif")

if __name__ == "__main__":
    create_training_animation()
    create_latent_space_animation()
