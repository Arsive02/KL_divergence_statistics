const TrainingDynamics = () => {

  return (
    <div className="space-y-8">
      <section>
        <h2 className="text-2xl font-serif text-gray-800 mb-4">Training Dynamics</h2>
        <p className="text-gray-700 mb-6">The training process revealed several interesting patterns across different model architectures.</p>
      </section>

      <section className="space-y-4">
        <h3 className="text-xl font-serif text-gray-800">Convergence Behavior</h3>
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <p className="text-gray-700 mb-4">Our analysis revealed distinct convergence patterns across the three models:</p>

          <div className="space-y-4">
            <div className="p-4 bg-blue-50 rounded-lg">
              <p className="text-gray-800">KL Divergence (P||Q) = Σ P(x) log(P(x) / Q(x))</p>
            </div>

            <div className="space-y-2">
              <p className="text-gray-700">
                <span className="font-semibold">Logistic Regression:</span> Demonstrated rapid initial convergence but quickly plateaued, suggesting limitations in capturing complex probability distributions.
              </p>
              <p className="text-gray-700">
                <span className="font-semibold">Random Forest:</span> Exhibited a more gradual improvement trajectory with better final convergence, leveraging its ensemble nature.
              </p>
              <p className="text-gray-700">
                <span className="font-semibold">Neural Network:</span> Emerged as the most effective model, achieving the lowest final KL divergence of 0.0000.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="space-y-4">
        <h3 className="text-xl font-serif text-gray-800">Distribution Matching</h3>
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <p className="text-gray-700 mb-4">
            The probability distribution analysis revealed significant differences in how each model approached the distribution matching task:
          </p>

          <div className="space-y-4">
            <div className="p-4 bg-gray-50 rounded-lg">
              <ul className="space-y-3">
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  <p className="text-gray-700">The neural network achieved the most faithful reproduction of the true probability distribution, particularly in capturing subtle variations in probability densities</p>
                </li>
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  <p className="text-gray-700">Random Forest tended to produce more discrete probability estimates, reflecting its underlying decision tree structure</p>
                </li>
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  <p className="text-gray-700">Logistic Regression showed good calibration but demonstrated less flexibility in capturing complex distributional patterns</p>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default TrainingDynamics;