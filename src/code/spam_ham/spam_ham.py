import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
from PIL import Image
import io
import seaborn as sns


class SpamClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SpamClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.sigmoid(self.layer3(x))
        return x
    
    
def prepare_data(data):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(data['text']).toarray()
    y = data['label'].values
    return X, y, vectorizer


def calculate_kl_divergence(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    kl_0 = (1 - y_true.mean()) * np.log((1 - y_true.mean()) / (1 - y_pred.mean() + epsilon))
    kl_1 = y_true.mean() * np.log(y_true.mean() / (y_pred.mean() + epsilon))
    return kl_0 + kl_1


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    return img


def create_training_animation():
    """
    Creates an animation of the training process for a spam classifier model.
    This function performs the following steps:
    1. Loads and prepares the data from 'ham_spam.csv'.
    2. Splits the data into training and validation sets.
    3. Converts the data into PyTorch tensors.
    4. Initializes the spam classifier model, loss function, and optimizer.
    5. Trains the model for a specified number of epochs, updating the model parameters.
    6. During each epoch, evaluates the model on the validation set and plots the true and predicted distributions.
    7. Calculates and plots the KL divergence between the true and predicted distributions over time.
    8. Converts the plots into images and compiles them into a GIF.
    The resulting animation is saved as 'training_animation.gif'.
    Note:
        - The function assumes the existence of the following helper functions:
            - `prepare_data(data)`: Prepares the data for training.
            - `SpamClassifier(input_dim)`: Initializes the spam classifier model.
            - `calculate_kl_divergence(y_true, y_pred)`: Calculates the KL divergence.
            - `fig2img(fig)`: Converts a matplotlib figure to an image.
    Returns:
        None
    """
    # Load and prepare data
    data = pd.read_csv('ham_spam.csv')
    X, y, vectorizer = prepare_data(data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
    
    model = SpamClassifier(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    frames = []
    n_epochs = 50
    kl_history = []
    
    # Create figure with 2 subplots (removing the middle plot)
    fig = plt.figure(figsize=(15, 6))
    gs = plt.GridSpec(1, 2, width_ratios=[1.5, 1])
    ax1 = fig.add_subplot(gs[0, 0])  # Combined distributions
    ax2 = fig.add_subplot(gs[0, 1])  # KL divergence
    
    # Generate synthetic points for true distribution KDE
    def generate_true_distribution(labels):
        # Add small random noise to create a distribution around 0 and 1
        noise_scale = 0.05
        true_probs = []
        for label in labels:
            if label == 0:
                noise = np.random.normal(0, noise_scale, 1)[0]
                true_probs.append(max(0, min(0.15, noise)))  # Clamp between 0 and 0.15
            else:
                noise = np.random.normal(0, noise_scale, 1)[0]
                true_probs.append(min(1, max(0.85, 1 + noise)))  # Clamp between 0.85 and 1
        return np.array(true_probs)
    
    # Generate true distribution points once
    true_distribution = generate_true_distribution(y_val)
    
    for epoch in tqdm(range(n_epochs)):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor).numpy()
            
            # Clear axes
            ax1.clear()
            ax2.clear()
            
            # Plot both distributions in the same plot
            # First, plot true distribution with dashed lines
            for label, color in zip([0, 1], ['blue', 'red']):
                mask = y_val == label
                if mask.any():
                    sns.kdeplot(
                        data=true_distribution[mask],
                        ax=ax1,
                        color=color,
                        linestyle='--',
                        label=f'True Class {label}',
                        fill=True,
                        alpha=0.15
                    )
            
            # Then, plot predicted distribution with solid lines
            for label, color in zip([0, 1], ['blue', 'red']):
                mask = y_val == label
                if mask.any():
                    sns.kdeplot(
                        data=val_outputs[mask].flatten(),
                        ax=ax1,
                        color=color,
                        linestyle='-',
                        label=f'Predicted Class {label}',
                        fill=True,
                        alpha=0.15
                    )
            
            ax1.set_title(f'Distribution Comparison (Epoch {epoch+1})')
            ax1.set_xlabel('Probability')
            ax1.set_ylabel('Density')
            ax1.set_xlim(0, 1)
            ax1.grid(True)
            # Adjust legend to show both true and predicted
            handles, labels = ax1.get_legend_handles_labels()
            # Reorder legend to group by class
            order = [0, 2, 1, 3]  # Reorder to group by class
            ax1.legend([handles[idx] for idx in order], 
                      [labels[idx] for idx in order])
            
            # Calculate and plot KL divergence
            kl_div = calculate_kl_divergence(y_val, val_outputs)
            kl_history.append(kl_div)
            
            ax2.plot(range(1, epoch + 2), kl_history, 'g-')
            ax2.set_title('KL Divergence Over Time')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('KL Divergence')
            ax2.set_xlim(0, n_epochs)
            ax2.grid(True)
            
            plt.tight_layout()
            
            # Convert plot to image and append to frames
            img = fig2img(fig)
            frames.append(img)
    
    plt.close()
    
    # Save as GIF
    frames[0].save(
        'training_animation.gif',
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=200,
        loop=0
    )
    
    print("Animation saved as training_animation.gif")

if __name__ == "__main__":
    create_training_animation()