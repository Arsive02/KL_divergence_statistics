// Model Performance Analysis Component
const ModelPerformance = () => {
  return (
    <div className="mb-8">
      <h2 className="text-2xl font-bold mb-6">Model Performance Analysis</h2>
      
      <div className="bg-white rounded-lg p-6 mb-6">
        <h3 className="text-xl font-semibold mb-4">Computational Efficiency</h3>
        <div className="space-y-4">
          <p>Logistic Regression proved to be the most computationally efficient, requiring minimal resources and achieving convergence in the shortest time.</p>
          <p>The Random Forest occupied a middle ground, benefiting from parallelizable computation that allowed for efficient scaling with available computing resources.</p>
          <p>The Neural Network, while achieving the best results, demanded the highest computational investment, requiring more iterations and longer training times to achieve convergence.</p>
        </div>
      </div>

      <div className="bg-white rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4">Scalability</h3>
        <div className="space-y-4">
          <p>Logistic Regression showed remarkable consistency, maintaining stable performance across different dataset sizes but with limited improvement potential.</p>
          <p>The Random Forest demonstrated more promising scaling characteristics, with performance improving significantly as more data became available.</p>
          <p>The Neural Network exhibited the most impressive scaling properties, with performance improving dramatically with increased data volume.</p>
        </div>
      </div>
    </div>
  );
};

// Discussion Component
const Discussion = () => {
  return (
    <div className="mb-8">
      <h2 className="text-2xl font-bold mb-6">Discussion</h2>
      
      <div className="bg-white rounded-lg p-6 mb-6">
        <h3 className="text-xl font-semibold mb-4">Insights from Binary Classification</h3>
        <div className="space-y-4">
          <p>Our experiments with the dataset provided compelling empirical evidence of the theoretical relationship between maximum likelihood estimation and KL divergence minimization.</p>
          <div className="p-4 bg-gray-50 rounded">
            <p>Log Likelihood is proportional to the negative KL Divergence</p>
          </div>
          <p>The opposing trajectories of log likelihood and KL divergence during training demonstrated their fundamental relationship, with improvements in one metric consistently corresponding to improvements in the other.</p>
        </div>
      </div>

      <div className="bg-white rounded-lg p-6 mb-6">
        <h3 className="text-xl font-semibold mb-4">Practical Implications</h3>
        <p>The relationship between maximum likelihood and KL divergence has significant practical implications for machine learning practitioners. Our analysis demonstrates that practitioners can choose between maximizing likelihood and minimizing KL divergence based on computational convenience.</p>
      </div>

      <div className="bg-white rounded-lg p-6">
        <h3 className="text-xl font-semibold mb-4">Future Directions</h3>
        <ul className="list-disc pl-6 space-y-2">
          <li>Investigation of how KL divergence behavior changes with different neural network architectures</li>
          <li>Development of hybrid approaches combining the strengths of different models</li>
          <li>Exploration of alternative divergence measures for probability distribution matching</li>
        </ul>
      </div>
    </div>
  );
};

// Conclusion Component
const Conclusion = () => {
  return (
    <div className="mb-8">
      <h2 className="text-2xl font-bold mb-6">Conclusion</h2>
      
      <div className="bg-white rounded-lg p-6 mb-6">
        <p className="mb-4">
          This study provides comprehensive empirical evidence for the relationship between maximum likelihood estimation and KL divergence minimization across different model architectures.
        </p>
        <p className="mb-4">
          Our results demonstrate that while all three models can effectively minimize KL divergence, they exhibit different convergence patterns and computational trade-offs.
        </p>
        <p>
          The neural network model achieved the lowest final KL divergence, suggesting superior capability in matching complex probability distributions. However, the simpler logistic regression and random forest models proved competitive, especially considering their computational efficiency.
        </p>
      </div>
    </div>
  );
};

// Export all components
export { ModelPerformance, Discussion, Conclusion };