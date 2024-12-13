import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Settings, Split, Timer } from 'lucide-react';

const TrainingProtocol: React.FC = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <h2 className="text-3xl font-serif text-slate-800 mb-6">Training Protocol</h2>

      <div className="grid gap-6">
        {/* Optimizer Settings */}
        <Card className="bg-white shadow-md">
          <CardHeader>
            <CardTitle className="flex items-center text-2xl text-slate-800">
              <Settings className="w-6 h-6 mr-2" />
              Optimizer Configuration
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3">
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                <span>Adam optimizer with learning rate of 0.001</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                <span>Training proceeded for 50 epochs for all models</span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                <span>Early stopping based on validation loss when applicable</span>
              </li>
            </ul>
          </CardContent>
        </Card>

        {/* Dataset Split */}
        <Card className="bg-white shadow-md">
          <CardHeader>
            <CardTitle className="flex items-center text-2xl text-slate-800">
              <Split className="w-6 h-6 mr-2" />
              Dataset Split
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex justify-center space-x-4 p-4">
              <div className="text-center p-4 bg-blue-100 rounded-lg flex-grow">
                <h4 className="font-semibold mb-2">Training Set</h4>
                <p className="text-2xl text-blue-600">80%</p>
              </div>
              <div className="text-center p-4 bg-green-100 rounded-lg flex-grow">
                <h4 className="font-semibold mb-2">Validation Set</h4>
                <p className="text-2xl text-green-600">20%</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* VAE Training */}
        <Card className="bg-white shadow-md">
          <CardHeader>
            <CardTitle className="flex items-center text-2xl text-slate-800">
              <Timer className="w-6 h-6 mr-2" />
              VAE Specific Training
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="bg-slate-50 p-4 rounded-lg">
              <p className="text-lg mb-4">
                For the Variational Autoencoder (VAE), we employed the standard ELBO 
                objective with two main components:
              </p>
              <ul className="space-y-3">
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  <span>Reconstruction loss using binary cross-entropy</span>
                </li>
                <li className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  <span>KL divergence terms for latent space regularization</span>
                </li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default TrainingProtocol;