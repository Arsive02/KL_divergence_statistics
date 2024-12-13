import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os


class BinaryClassifier(nn.Module):
    """
    A binary classifier neural network using PyTorch's nn.Module.
    Args:
        input_size (int): The number of input features.
    Attributes:
        layer1 (nn.Linear): The first linear layer.
        layer2 (nn.Linear): The second linear layer.
        layer3 (nn.Linear): The third linear layer.
        relu (nn.ReLU): The ReLU activation function.
        sigmoid (nn.Sigmoid): The Sigmoid activation function.
    Methods:
        forward(x):
            Defines the forward pass of the network.
            Args:
                x (torch.Tensor): The input tensor.
            Returns:
                torch.Tensor: The output tensor after passing through the network.
    """
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x
    

def compute_kl_divergence(p_density, q_density, x_eval):
    p = np.clip(p_density(x_eval), 1e-10, None)
    q = np.clip(q_density(x_eval), 1e-10, None)
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q)) * (x_eval[1] - x_eval[0])


# Setup
os.makedirs('kl_viz', exist_ok=True)
x_eval = np.linspace(0, 1, 1000)

# Load and preprocess data
data = pd.read_csv('lung_cancer.csv')
X = data.drop(['LUNG_CANCER'], axis=1)
y = (data['LUNG_CANCER'] == 'YES').astype(int)
X = pd.get_dummies(X, columns=['GENDER'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train.values)

# Initialize model
model = BinaryClassifier(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training and visualization
kl_values = []
n_epochs = 1000
save_interval = 100

for epoch in range(n_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs.squeeze(), y_train_tensor)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % save_interval == 0:
        # Get predictions
        with torch.no_grad():
            preds = model(X_train_tensor).numpy()
        
        # Estimate PDFs
        true_density = gaussian_kde(y_train.values.reshape(-1))
        pred_density = gaussian_kde(preds.reshape(-1))
        
        # Calculate KL divergence
        kl_div = compute_kl_divergence(true_density, pred_density, x_eval)
        kl_values.append(kl_div)
        
        # Create plot
        fig = plt.figure(figsize=(15, 8))
        fig.patch.set_facecolor('white')
        
        # Plot PDFs
        ax1 = plt.subplot(1, 2, 1)
        ax1.plot(x_eval, true_density(x_eval), 'b-', label='True Distribution', linewidth=2)
        ax1.plot(x_eval, pred_density(x_eval), 'r--', label='Neural Network', linewidth=2)
        ax1.set_xlabel('Probability')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Probability Density Functions\nEpoch {epoch + 1}')
        ax1.legend()
        ax1.grid(True)
        
        # Plot KL divergence trend
        ax2 = plt.subplot(1, 2, 2)
        ax2.plot(range(save_interval, epoch + save_interval + 1, save_interval), 
                kl_values, 'b-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('KL Divergence')
        ax2.set_title('KL Divergence Over Time')
        ax2.grid(True)
        
        plt.tight_layout(pad=2.0)
        plt.savefig(f'kl_viz/nn_epoch_{epoch + 1:04d}.png', 
                    bbox_inches='tight', 
                    dpi=100)
        plt.close()

# Create animation
images = []
for i in range(save_interval, n_epochs + 1, save_interval):
    filename = f'kl_viz/nn_epoch_{i:04d}.png'
    images.append(imageio.imread(filename))

imageio.mimsave('neural_network_training.gif', images, duration=1.0, loop=0)

print(f"Final KL Divergence - Neural Network: {kl_values[-1]:.4f}")