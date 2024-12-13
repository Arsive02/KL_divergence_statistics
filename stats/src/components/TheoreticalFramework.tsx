import { Card, CardContent } from "@/components/ui/card";

const TheoreticalFramework = () => {
  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Title Section */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-serif text-slate-800">Theoretical Framework</h1>
      </div>

      {/* Maximum Likelihood Connection */}
      <Card className="bg-white shadow-md">
        <CardContent className="p-6">
          <h2 className="text-2xl font-serif mb-4 text-slate-800 border-b border-slate-200 pb-2">
            Connection to Maximum Likelihood Estimation
          </h2>
          <div className="prose prose-slate max-w-none">
            <p className="text-lg leading-relaxed mb-4">
              The connection between KL divergence and maximum likelihood estimation (MLE) provides 
              a theoretical foundation for many machine learning objectives. Consider a parametric 
              model qθ(x) trying to approximate the true data distribution p(x).
            </p>
            <div className="bg-slate-50 p-4 rounded-lg my-4">
              <p className="text-lg mb-2">The MLE objective is to maximize:</p>
              <p className="text-lg font-mono">L(θ) = E<sub>x~p(x)</sub>[log q<sub>θ</sub>(x)]</p>
              <p className="text-lg mt-4 mb-2">This is equivalent to minimizing:</p>
              <p className="text-lg font-mono">-L(θ) = -E<sub>x~p(x)</sub>[log q<sub>θ</sub>(x)] = E<sub>x~p(x)</sub>[-log q<sub>θ</sub>(x)]</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* KL Divergence and Cross-Entropy */}
      <Card className="bg-white shadow-md">
        <CardContent className="p-6">
          <h2 className="text-2xl font-serif mb-4 text-slate-800 border-b border-slate-200 pb-2">
            Relationship between KL Divergence and Cross-Entropy
          </h2>
          <div className="prose prose-slate max-w-none">
            <p className="text-lg leading-relaxed mb-4">
              The relationship between KL divergence and cross-entropy is fundamental to understanding 
              why cross-entropy is commonly used as a loss function in machine learning.
            </p>
            <div className="bg-slate-50 p-4 rounded-lg my-4">
              <p className="text-lg font-mono">D<sub>KL</sub>(P||Q) = Σ<sub>x</sub> P(x) log (P(x) / Q(x))</p>
              <p className="text-lg font-mono">= Σ<sub>x</sub> P(x) log P(x) - Σ<sub>x</sub> P(x) log Q(x)</p>
              <p className="text-lg font-mono">= -H(P) + H(P,Q)</p>
            </div>
            <p className="text-lg leading-relaxed">
              where H(P) is the entropy of distribution P, and H(P,Q) is the cross-entropy between P and Q.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Mathematical Properties */}
      <Card className="bg-white shadow-md">
        <CardContent className="p-6">
          <h2 className="text-2xl font-serif mb-4 text-slate-800 border-b border-slate-200 pb-2">
            Key Mathematical Properties
          </h2>
          <div className="prose prose-slate max-w-none">
            <ul className="space-y-4">
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">1.</span>
                <div>
                  <span className="font-semibold">Non-negativity:</span>
                  <p className="text-lg">
                    D<sub>KL</sub>(P||Q) &ge; 0 for all distributions P and Q, with equality if and only 
                    if P = Q almost everywhere.
                  </p>
                </div>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">2.</span>
                <div>
                  <span className="font-semibold">Asymmetry:</span>
                  <p className="text-lg">
                    D<sub>KL</sub>(P||Q) &ne; D<sub>KL</sub>(Q||P) in general, i.e., KL divergence is not 
                    symmetric.
                  </p>
                </div>
              </li>
              <li className="flex items-start">
                <span className="text-blue-600 mr-2">3.</span>
                <div>
                  <span className="font-semibold">Chain Rule:</span>
                  <p className="text-lg">
                    For joint distributions:
                    D<sub>KL</sub>(P(X,Y)||Q(X,Y)) = D<sub>KL</sub>(P(X)||Q(X)) + D<sub>KL</sub>(P(Y|X)||Q(Y|X))
                  </p>
                </div>
              </li>
            </ul>
          </div>
        </CardContent>
      </Card>

      {/* Binary Classification Context */}
      <Card className="bg-white shadow-md">
        <CardContent className="p-6">
          <h2 className="text-2xl font-serif mb-4 text-slate-800 border-b border-slate-200 pb-2">
            KL Divergence in Binary Classification
          </h2>
          <div className="prose prose-slate max-w-none">
            <p className="text-lg leading-relaxed mb-4">
              In binary classification, KL divergence forms the theoretical foundation of the commonly 
              used cross-entropy loss function.
            </p>
            <div className="bg-slate-50 p-4 rounded-lg my-4">
              <p className="text-lg mb-2">For a single binary classification example:</p>
              <p className="text-lg font-mono">D<sub>KL</sub>(p₁||q₁) = p₁ log(p₁ / q₁) + (1 - p₁) log((1 - p₁) / (1 - q₁))</p>
            </div>
            <p className="text-lg leading-relaxed">
              where p₁ is the true probability (usually 0 or 1) and q₁ is the model's predicted probability.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default TheoreticalFramework;