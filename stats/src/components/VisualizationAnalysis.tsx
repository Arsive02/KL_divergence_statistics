import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Play, LineChart, Network, Binary } from 'lucide-react';

const VisualizationAnalysis: React.FC = () => {
  return (
    <div className="max-w-6xl mx-auto space-y-8">
    {/* Original Static Visualizations */}
      <Card className="bg-white shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center text-2xl text-slate-800">
            <LineChart className="w-6 h-6 mr-2" />
            Training Dynamics and Performance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <img 
                src="convergence_plot.png" 
                alt="Convergence Plot"
                className="w-full rounded-lg shadow-md"
              />
              <p className="text-sm text-slate-600 text-center">
                Convergence of Log Likelihood and KL Divergence over iterations
              </p>
            </div>
            <div className="space-y-2">
              <img 
                src="probability_dist.png" 
                alt="Probability Distribution"
                className="w-full rounded-lg shadow-md"
              />
              <p className="text-sm text-slate-600 text-center">
                Distribution of predicted probabilities using KDE
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
      {/* Traditional Models Animation */}
      <Card className="bg-white shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center text-2xl text-slate-800">
            <Play className="w-6 h-6 mr-2" />
            Traditional Models Training Evolution
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="bg-slate-50 rounded-lg overflow-hidden p-4">
              <img 
                src="model_comparison.gif" 
                alt="Traditional Models Training Animation"
                className="w-full max-h-[600px] object-contain rounded-lg mx-auto animate-slow"
                style={{ animationDuration: '3s' }}
              />
            </div>
            <p className="text-lg text-slate-600 text-center">
              Visualization of training evolution for Logistic Regression and Random Forest models,
              showing the progressive improvement in distribution matching over iterations.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Neural Network Animation */}
      <Card className="bg-white shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center text-2xl text-slate-800">
            <Network className="w-6 h-6 mr-2" />
            Neural Network Training Progression
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="bg-slate-50 rounded-lg overflow-hidden p-4">
              <img 
                src="neural_network_training.gif" 
                alt="Neural Network Training Animation"
                className="w-full max-h-[600px] object-contain rounded-lg mx-auto animate-slow"
                style={{ animationDuration: '3s' }}
              />
            </div>
            <p className="text-lg text-slate-600 text-center">
              Neural Network training progression showing convergence behavior and 
              distribution matching capabilities over multiple epochs.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Spam Classifier Animation */}
      <Card className="bg-white shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center text-2xl text-slate-800">
            <Binary className="w-6 h-6 mr-2" />
            Spam Classifier Training Dynamics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="bg-slate-50 rounded-lg overflow-hidden p-4">
              <img 
                src="training_animation.gif" 
                alt="Spam Classifier Training Animation"
                className="w-full max-h-[600px] object-contain rounded-lg mx-auto animate-slow"
                style={{ animationDuration: '3s' }}
              />
            </div>
            <p className="text-lg text-slate-600 text-center">
              Visualization of the spam classifier's training process, demonstrating
              the evolution of class separation and decision boundaries.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Latent Space Animation */}
      <Card className="bg-white shadow-md">
        <CardHeader>
          <CardTitle className="flex items-center text-2xl text-slate-800">
            <Network className="w-6 h-6 mr-2" />
            VAE Latent Space Evolution
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="bg-slate-50 rounded-lg overflow-hidden p-4">
              <img 
                src="latent_space_evolution.gif" 
                alt="Latent Space Evolution"
                className="w-full max-h-[600px] object-contain rounded-lg mx-auto animate-slow"
                style={{ animationDuration: '3s' }}
              />
            </div>
            <p className="text-lg text-slate-600 text-center">
              Evolution of the VAE latent space during training, showing the emergence
              of meaningful representations and cluster formation.
            </p>
          </div>
        </CardContent>
      </Card>
      <style >{`
        .animate-slow {
          animation-duration: 3s !important;
          animation-timing-function: linear !important;
        }
      `}</style>
    </div>
  );
};

export default VisualizationAnalysis;