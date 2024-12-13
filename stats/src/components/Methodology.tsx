import React from 'react';

const Methodology: React.FC = () => (
  <div>
    <h2>Experimental Setup</h2>
    <div className="methodology-section">
      <div className="methodology-card">
        <h3>Datasets Used</h3>
        <ul>
          <li>Lung Cancer Dataset - Medical records with 15 features</li>
          <li>Spam Classification Dataset - Ham/spam email text data</li>
          <li>MNIST Dataset - Handwritten digits for VAE analysis</li>
          <li>Abalone Dataset - Physical measurements for binary classification</li>
        </ul>
      </div>
      <div className="methodology-card">
        <h3>Model Architectures</h3>
        <ul>
          <li>Logistic Regression with L2 regularization</li>
          <li>Random Forest with 100 trees</li>
          <li>Neural Network with three layers (input → 64 → 32 → 1)</li>
        </ul>
      </div>
    </div>
  </div>
);

export default Methodology;