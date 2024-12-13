import { Card, CardContent } from "@/components/ui/card";

const Introduction = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Title Section */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-serif mb-4 text-slate-800">Kullback-Leibler Divergence: A Statistical Bridge</h1>
        <p className="text-xl text-slate-600">Between Information Theory and Machine Learning</p>
      </div>

      {/* Background and Motivation */}
      <Card className="bg-white shadow-md">
        <CardContent className="p-6">
          <h2 className="text-2xl font-serif mb-4 text-slate-800 border-b border-slate-200 pb-2">
            Background and Motivation
          </h2>
          <div className="prose prose-slate max-w-none">
            <p className="text-lg leading-relaxed mb-6">
              The Kullback-Leibler (KL) divergence, also known as relative entropy, is one of the most 
              fundamental concepts bridging information theory and statistical inference. Originally 
              introduced by Solomon Kullback and Richard Leibler in 1951, this measure has become 
              increasingly relevant in modern statistical applications, particularly in machine learning 
              and Bayesian inference.
            </p>
            <p className="text-lg leading-relaxed">
              In its essence, KL divergence quantifies the difference between two probability distributions 
              on the same random variable. Unlike traditional metrics, it is not symmetric and does not 
              satisfy the triangle inequality, yet these very properties make it uniquely suited for many 
              statistical applications.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Research Objectives */}
      <Card className="bg-white shadow-md">
        <CardContent className="p-6">
          <h2 className="text-2xl font-serif mb-4 text-slate-800 border-b border-slate-200 pb-2">
            Research Objectives
          </h2>
          <div className="prose prose-slate max-w-none">
            <p className="text-lg leading-relaxed mb-4">
              This research aims to provide comprehensive insights into the practical applications and 
              theoretical foundations of KL divergence in modern machine learning.
            </p>
            <ul className="space-y-3">
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                <span className="text-lg">
                  Visualize and analyze the evolution of KL divergence during the training of three 
                  different binary classification models
                </span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                <span className="text-lg">
                  Compare how different model architectures affect the behavior of KL divergence
                </span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                <span className="text-lg">
                  Demonstrate the role of KL divergence in Variational Autoencoders
                </span>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">•</span>
                <span className="text-lg">
                  Provide practical insights into the relationship between KL divergence and model 
                  performance
                </span>
              </li>
            </ul>
          </div>
        </CardContent>
      </Card>

      {/* Mathematical Foundation */}
      <Card className="bg-white shadow-md">
        <CardContent className="p-6">
          <h2 className="text-2xl font-serif mb-4 text-slate-800 border-b border-slate-200 pb-2">
            Mathematical Foundation
          </h2>
          <div className="prose prose-slate max-w-none">
            <p className="text-lg leading-relaxed mb-4">
              For discrete probability distributions P and Q, the KL divergence is defined as:
            </p>
            <div className="bg-slate-50 p-4 rounded-lg text-center my-4 font-serif">
              <span className="text-lg">KL(P || Q) = ∑ P(x) log(P(x) / Q(x))</span>
            </div>
            <p className="text-lg leading-relaxed">
              This measure provides a fundamental way to assess how one probability distribution 
              differs from another, serving as a cornerstone for various machine learning algorithms 
              and theoretical analyses.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Introduction;