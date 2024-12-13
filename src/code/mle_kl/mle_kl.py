import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_curve, auc, calibration_curve
import matplotlib.pyplot as plt
from scipy.special import xlogy
import seaborn as sns

# Data Preparation
data = pd.read_csv("data\\abalone.csv")
data['BinaryTarget'] = (data['Type'] == 'M').astype(int)

X = data[['LongestShell', 'Diameter', 'Height', 'WholeWeight', 
          'ShuckedWeight', 'VisceraWeight', 'ShellWeight']]
y = data['BinaryTarget']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Logistic Regression (MLE) model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_scaled, y)
y_pred_prob_mle = log_reg.predict_proba(X_scaled)[:, 1]

# Train SGD Classifier model
sgd = SGDClassifier(loss="log_loss", max_iter=1000)
sgd.fit(X_scaled, y)
y_pred_prob_sgd = sgd.predict_proba(X_scaled)[:, 1]

# Plotting
plt.figure(figsize=(15, 10))

# Plot 1: Histogram of probabilities
plt.subplot(2, 2, 1)
plt.hist(y_pred_prob_mle, bins=30, alpha=0.5, label='MLE Predictions', density=True)
plt.hist(y_pred_prob_sgd, bins=30, alpha=0.5, label='SGD Predictions', density=True)
plt.axvline(x=0.5, color='r', linestyle='--', label='Decision Boundary')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Distribution of Predicted Probabilities')
plt.legend()

# Plot 2: KDE plot
plt.subplot(2, 2, 2)
sns.kdeplot(data=y_pred_prob_mle, label='MLE Predictions')
sns.kdeplot(data=y_pred_prob_sgd, label='SGD Predictions')
plt.axvline(x=0.5, color='r', linestyle='--', label='Decision Boundary')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('KDE of Predicted Probabilities')
plt.legend()

# Plot 3: Prediction distributions by true class
plt.subplot(2, 2, 3)
for target_class in [0, 1]:
    mask = y == target_class
    plt.hist(y_pred_prob_mle[mask], bins=20, alpha=0.5, 
             label=f'True Class {target_class}', density=True)
plt.axvline(x=0.5, color='r', linestyle='--', label='Decision Boundary')
plt.xlabel('MLE Predicted Probability')
plt.ylabel('Density')
plt.title('MLE Predictions by True Class')
plt.legend()

# Plot 4: ROC curves
plt.subplot(2, 2, 4)
# ROC for MLE
fpr_mle, tpr_mle, _ = roc_curve(y, y_pred_prob_mle)
roc_auc_mle = auc(fpr_mle, tpr_mle)
plt.plot(fpr_mle, tpr_mle, label=f'MLE (AUC = {roc_auc_mle:.2f})')

# ROC for SGD
fpr_sgd, tpr_sgd, _ = roc_curve(y, y_pred_prob_sgd)
roc_auc_sgd = auc(fpr_sgd, tpr_sgd)
plt.plot(fpr_sgd, tpr_sgd, label=f'SGD (AUC = {roc_auc_sgd:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()

plt.tight_layout()
plt.show()

# Summary Statistics
print("\nSummary Statistics:")
print(f"MLE predictions mean: {np.mean(y_pred_prob_mle):.3f}")
print(f"MLE predictions std: {np.std(y_pred_prob_mle):.3f}")
print(f"SGD predictions mean: {np.mean(y_pred_prob_sgd):.3f}")
print(f"SGD predictions std: {np.std(y_pred_prob_sgd):.3f}")

# Calibration Metrics
prob_true_mle, prob_pred_mle = calibration_curve(y, y_pred_prob_mle, n_bins=10)
prob_true_sgd, prob_pred_sgd = calibration_curve(y, y_pred_prob_sgd, n_bins=10)

print("\nCalibration Metrics:")
print("MLE predicted probabilities vs true proportions:")
for true, pred in zip(prob_true_mle, prob_pred_mle):
    print(f"Predicted: {pred:.3f}, Actual: {true:.3f}")

print("\nSGD predicted probabilities vs true proportions:")
for true, pred in zip(prob_true_sgd, prob_pred_sgd):
    print(f"Predicted: {pred:.3f}, Actual: {true:.3f}")

# Define metrics computation functions
def compute_log_likelihood(y_true, y_pred_prob):
    """Compute log likelihood with numerical stability"""
    eps = 1e-15
    y_pred_prob = np.clip(y_pred_prob, eps, 1 - eps)
    return abs(np.sum(y_true * np.log(y_pred_prob) + (1 - y_true) * np.log(1 - y_pred_prob)))

def compute_kl_divergence(p_true, p_pred):
    """Compute KL divergence with numerical stability"""
    eps = 1e-15
    p_pred = np.clip(p_pred, eps, 1 - eps)
    p_true = np.clip(p_true, eps, 1 - eps)
    return np.sum(xlogy(p_true, p_true/p_pred) + xlogy(1 - p_true, (1 - p_true)/(1 - p_pred)))

# Train models and collect metrics across epochs
n_iterations = [1, 2, 5, 8, 10, 20]
mle_results = []
kl_results = []
sgd_results = []

for n_iter in n_iterations:
    # MLE (Logistic Regression)
    log_reg = LogisticRegression(max_iter=n_iter)
    log_reg.fit(X_scaled, y)
    y_pred_mle = log_reg.predict_proba(X_scaled)[:, 1]
    
    # SGD Classifier
    sgd = SGDClassifier(loss="log_loss", max_iter=n_iter, learning_rate='optimal')
    sgd.fit(X_scaled, y)
    y_pred_sgd = sgd.predict_proba(X_scaled)[:, 1]
    
    # Compute metrics
    ll = compute_log_likelihood(y, y_pred_mle)
    kl = compute_kl_divergence(y, y_pred_mle)
    sgd_ll = compute_log_likelihood(y, y_pred_sgd)
    
    mle_results.append(ll)
    kl_results.append(kl)
    sgd_results.append(sgd_ll)

# Create figure with multiple subplots
fig_size = (6, 5)

# Plot 1: Convergence of MLE and KL Divergence
plt.figure(figsize=fig_size)
plt.plot(n_iterations, mle_results, 'b-o', label='Log Likelihood (MLE)')
plt.plot(n_iterations, kl_results, 'r-o', label='KL Divergence')
plt.plot(n_iterations, sgd_results, 'g-o', label='Log Likelihood (SGD)')
plt.xlabel('Iterations')
plt.ylabel('Metric Value')
plt.title('Convergence Analysis')
plt.legend()
plt.xscale('log')
plt.grid(True)
plt.tight_layout()
plt.savefig('mle_Kl\\convergence_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Distribution of probabilities (Histogram)
plt.figure(figsize=fig_size)
sns.kdeplot(data=y_pred_prob_mle, label='MLE Predictions', fill=True)
sns.kdeplot(data=y_pred_prob_sgd, label='SGD Predictions', fill=True)
plt.axvline(x=0.5, color='r', linestyle='--', label='Decision Boundary')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Probability Distributions')
plt.legend()
plt.tight_layout()
plt.savefig('mle_Kl\\probability_dist.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Class distribution
plt.figure(figsize=fig_size)
for target_class in [0, 1]:
    mask = y == target_class
    sns.kdeplot(data=y_pred_prob_mle[mask], 
                label=f'Class {target_class}', 
                fill=True)
plt.axvline(x=0.5, color='r', linestyle='--', label='Decision Boundary')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.title('Class-wise Distributions')
plt.legend()
plt.tight_layout()
plt.savefig('mle_Kl\\class_dist.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: ROC Curves
plt.figure(figsize=fig_size)
fpr_mle, tpr_mle, _ = roc_curve(y, y_pred_prob_mle)
roc_auc_mle = auc(fpr_mle, tpr_mle)
plt.plot(fpr_mle, tpr_mle, label=f'MLE (AUC = {roc_auc_mle:.2f})')

fpr_sgd, tpr_sgd, _ = roc_curve(y, y_pred_prob_sgd)
roc_auc_sgd = auc(fpr_sgd, tpr_sgd)
plt.plot(fpr_sgd, tpr_sgd, label=f'SGD (AUC = {roc_auc_sgd:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('mle_Kl\\roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# Print final metrics
print("\nFinal Metrics:")
print(f"MLE Log Likelihood: {mle_results[-1]:.3f}")
print(f"MLE KL Divergence: {kl_results[-1]:.3f}")
print(f"SGD Log Likelihood: {sgd_results[-1]:.3f}")

# Calculate and print calibration metrics
prob_true_mle, prob_pred_mle = calibration_curve(y, y_pred_prob_mle, n_bins=10)

print("\nCalibration Metrics (MLE):")
for true, pred in zip(prob_true_mle, prob_pred_mle):
    print(f"Predicted: {pred:.3f}, Actual: {true:.3f}")
