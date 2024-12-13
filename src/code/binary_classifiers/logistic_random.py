import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

def compute_kl_divergence(p_density, q_density, x_eval):
    """
    Compute the Kullback-Leibler (KL) divergence between two probability density functions.

    Parameters:
    p_density (function): A function that computes the probability density of distribution P.
    q_density (function): A function that computes the probability density of distribution Q.
    x_eval (numpy.ndarray): An array of points at which to evaluate the densities.

    Returns:
    float: The KL divergence between the two distributions.

    Notes:
    - The KL divergence is computed as the sum of p(x) * log(p(x) / q(x)) over all x in x_eval.
    - The densities are clipped to avoid division by zero and normalized to ensure they sum to 1.
    - The result is scaled by the spacing between points in x_eval to approximate the integral.
    """
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

# Initialize models
lr_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier(n_estimators=100)

# Training and visualization
kl_values_lr = []
kl_values_rf = []

for i in range(10):  # 10 iterations
    subset_size = len(X_train) // 10 * (i + 1)
    X_subset = X_train[:subset_size]
    y_subset = y_train[:subset_size]
    
    # Train models
    lr_model.fit(X_subset, y_subset)
    rf_model.fit(X_subset, y_subset)
    
    lr_preds = lr_model.predict_proba(X_train)[:, 1]
    rf_preds = rf_model.predict_proba(X_train)[:, 1]
    
    # Estimate PDFs
    true_density = gaussian_kde(np.array(y_train).reshape(-1))
    lr_density = gaussian_kde(np.array(lr_preds).reshape(-1))
    rf_density = gaussian_kde(np.array(rf_preds).reshape(-1))
    
    # Calculate KL divergence
    kl_values_lr.append(compute_kl_divergence(true_density, lr_density, x_eval))
    kl_values_rf.append(compute_kl_divergence(true_density, rf_density, x_eval))
    
    # Create plot with fixed size
    fig = plt.figure(figsize=(15, 8))
    fig.patch.set_facecolor('white')
    
    # Plot PDFs
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(x_eval, true_density(x_eval), 'b-', label='True Distribution', linewidth=2)
    ax1.plot(x_eval, lr_density(x_eval), 'r--', label='Logistic Regression', linewidth=2)
    ax1.plot(x_eval, rf_density(x_eval), 'g--', label='Random Forest', linewidth=2)
    ax1.set_xlabel('Probability')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Probability Density Functions\nIteration {i+1}')
    ax1.legend()
    ax1.grid(True)
    
    # Plot KL divergence trends
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(range(1, i+2), kl_values_lr, 'r-', label='Logistic Regression', linewidth=2)
    ax2.plot(range(1, i+2), kl_values_rf, 'g-', label='Random Forest', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('KL Divergence')
    ax2.set_title('KL Divergence Over Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(f'kl_viz/comparison_iteration_{i+1:02d}.png', 
                bbox_inches='tight', 
                dpi=100)
    plt.close()

# Create animation
images = []
for i in range(1, 11):
    filename = f'kl_viz/comparison_iteration_{i:02d}.png'
    images.append(imageio.imread(filename))

imageio.mimsave('model_comparison.gif', images, duration=1.0, loop=0)

# Print final KL divergence values
print(f"Final KL Divergence - Logistic Regression: {kl_values_lr[-1]:.4f}")
print(f"Final KL Divergence - Random Forest: {kl_values_rf[-1]:.4f}")