import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Database, Download } from 'lucide-react';

const Methodology: React.FC = () => (
  <div className="max-w-4xl mx-auto space-y-8">
    <h2 className="text-3xl font-serif text-slate-800 mb-6">Experimental Setup</h2>
    
    {/* Datasets Section */}
    <Card className="bg-white shadow-md">
      <CardHeader>
        <CardTitle className="flex items-center text-2xl text-slate-800">
          <Database className="w-6 h-6 mr-2" />
          Datasets Used
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="space-y-4">
          <li className="flex items-start">
            <span className="text-blue-600 mr-2">•</span>
            <div>
              <span className="font-semibold">Lung Cancer Dataset</span>
              <p className="text-slate-600">Medical records with 15 features including patient demographics and symptoms</p>
            </div>
          </li>
          <li className="flex items-start">
            <span className="text-blue-600 mr-2">•</span>
            <div>
              <span className="font-semibold">Spam Classification Dataset</span>
              <p className="text-slate-600">Ham/spam email text data for natural language processing analysis</p>
            </div>
          </li>
          <li className="flex items-start">
            <span className="text-blue-600 mr-2">•</span>
            <div>
              <span className="font-semibold">MNIST Dataset</span>
              <p className="text-slate-600">Handwritten digits for VAE analysis and visualization</p>
            </div>
          </li>
          <li className="flex items-start">
            <span className="text-blue-600 mr-2">•</span>
            <div>
              <span className="font-semibold">Abalone Dataset</span>
              <p className="text-slate-600">Physical measurements for binary classification experiments</p>
            </div>
          </li>
        </ul>

        <div className="mt-6 p-4 bg-blue-50 rounded-lg">
          <a 
            href="https://drive.google.com/drive/folders/1v5V6oQSznilQLaszxxOB_JplZwhS1Zms?usp=sharing"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-center text-blue-600 hover:text-blue-800 transition-colors duration-200"
          >
            <Download className="w-5 h-5 mr-2" />
            <span className="font-medium">Download Datasets</span>
          </a>
        </div>
      </CardContent>
    </Card>
  </div>
);

export default Methodology;