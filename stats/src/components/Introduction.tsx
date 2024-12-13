import React from 'react';

const Introduction: React.FC = () => (
  <div>
    <h2>Background and Motivation</h2>
    <p>
      The Kullback-Leibler (KL) divergence, also known as relative entropy, is one of the most fundamental concepts bridging information theory and statistical inference. Originally introduced by Solomon Kullback and Richard Leibler in 1951, this measure has become increasingly relevant in modern statistical applications, particularly in machine learning and Bayesian inference.
    </p>
    <h2>Research Objectives</h2>
    <ul>
      <li>Visualize and analyze the evolution of KL divergence during the training of three different binary classification models</li>
      <li>Compare how different model architectures affect the behavior of KL divergence</li>
      <li>Demonstrate the role of KL divergence in Variational Autoencoders</li>
      <li>Provide practical insights into the relationship between KL divergence and model performance</li>
    </ul>
  </div>
);

export default Introduction;