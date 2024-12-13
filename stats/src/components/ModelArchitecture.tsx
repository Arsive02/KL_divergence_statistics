import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Network, GitBranch, Box } from 'lucide-react';

const ModelArchitectures: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <h2 className="text-3xl font-serif text-slate-800 mb-6">Model Architectures</h2>
      <p className="text-lg text-slate-600 mb-8">
        We implemented multiple model architectures to compare how different approaches 
        affect the distribution matching process.
      </p>

      {/* Binary Classification Models */}
      <Card className="bg-white shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center text-2xl text-slate-800">
            <Network className="w-6 h-6 mr-2" />
            Binary Classification Models
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <p className="text-slate-600">
            For the lung cancer prediction task, we implemented three distinct architectures:
          </p>

          <div className="grid gap-6">
            {/* Logistic Regression */}
            <div className="p-4 bg-slate-50 rounded-lg">
              <h4 className="text-lg font-semibold text-slate-800 mb-2">Logistic Regression</h4>
              <ul className="list-disc pl-6 space-y-2 text-slate-600">
                <li>Linear model serving as baseline</li>
                <li>L2 regularization for parameter control</li>
                <li>Optimized with stochastic gradient descent</li>
              </ul>
            </div>

            {/* Random Forest */}
            <div className="p-4 bg-slate-50 rounded-lg">
              <h4 className="text-lg font-semibold text-slate-800 mb-2">Random Forest</h4>
              <ul className="list-disc pl-6 space-y-2 text-slate-600">
                <li>Ensemble model with 100 trees</li>
                <li>Non-linear decision boundaries</li>
                <li>Naturally bounded probability estimates</li>
              </ul>
            </div>

            {/* Neural Network */}
            <div className="p-4 bg-slate-50 rounded-lg">
              <h4 className="text-lg font-semibold text-slate-800 mb-2">Neural Network</h4>
              <ul className="list-disc pl-6 space-y-2 text-slate-600">
                <li>Three-layer architecture: input → 64 → 32 → 1</li>
                <li>ReLU activations for non-linearity</li>
                <li>Dropout (0.3) for regularization</li>
                <li>Sigmoid output layer for probability estimation</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Spam Classification Model */}
      <Card className="bg-white shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center text-2xl text-slate-800">
            <GitBranch className="w-6 h-6 mr-2" />
            Spam Classification Model
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <p className="text-slate-600">
            For text classification, we employed a neural network architecture specifically 
            designed for high-dimensional sparse input:
          </p>

          <div className="p-4 bg-slate-50 rounded-lg">
            <ul className="space-y-3 text-slate-600">
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                Input layer matching TF-IDF dimensionality (1,000)
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                Two hidden layers (128 and 64 units) with ReLU activation
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                Dropout layers (0.3) for regularization
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                Sigmoid output layer for binary classification
              </li>
            </ul>
          </div>
        </CardContent>
      </Card>

      {/* Variational Autoencoder */}
      <Card className="bg-white shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center text-2xl text-slate-800">
            <Box className="w-6 h-6 mr-2" />
            Variational Autoencoder
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <p className="text-slate-600">
            For the MNIST dataset, we implemented a VAE with the following structure:
          </p>

          <div className="grid gap-4">
            {/* Encoder */}
            <div className="p-4 bg-slate-50 rounded-lg">
              <h4 className="text-lg font-semibold text-slate-800 mb-2">Encoder</h4>
              <p className="text-slate-600">
                Two fully connected layers (784 → 400 → 400) with ReLU activation
              </p>
            </div>

            {/* Latent Space */}
            <div className="p-4 bg-slate-50 rounded-lg">
              <h4 className="text-lg font-semibold text-slate-800 mb-2">Latent Space</h4>
              <p className="text-slate-600">
                2-dimensional representation with separate networks for mean and 
                log-variance estimation
              </p>
            </div>

            {/* Decoder */}
            <div className="p-4 bg-slate-50 rounded-lg">
              <h4 className="text-lg font-semibold text-slate-800 mb-2">Decoder</h4>
              <p className="text-slate-600">
                Mirror of the encoder architecture (2 → 400 → 400 → 784) with 
                sigmoid output activation
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default ModelArchitectures;