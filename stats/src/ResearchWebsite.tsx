import { useState } from 'react';
import { Book, Users, FileText } from 'lucide-react';
import './index.css';

type SectionKey = 'introduction' | 'authors' | 'abstract' | 'methodology';

const ResearchWebsite = () => {
  const [activeSection, setActiveSection] = useState<SectionKey>('introduction');
  
  const sections: Record<SectionKey, { title: string; icon: React.ComponentType; content: JSX.Element }> = {
    introduction: {
      title: 'Introduction',
      icon: Book,
      content: (
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
      )
    },
    authors: {
      title: 'Authors',
      icon: Users,
      content: (
        <div className="authors-grid">
          <div className="author-card">
            <h3>Sivakumar Ramakrishnan</h3>
            <p>Department of Data Science</p>
            <p>University of Colorado Boulder</p>
          </div>
          <div className="author-card">
            <h3>Sai Nandini Tata</h3>
            <p>Department of Data Science</p>
            <p>University of Colorado Boulder</p>
          </div>
          <div className="author-card">
            <h3>Deep Shukla</h3>
            <p>Department of Data Science</p>
            <p>University of Colorado Boulder</p>
          </div>
        </div>
      )
    },
    abstract: {
      title: 'Abstract',
      icon: FileText,
      content: (
        <div className="abstract-card">
          <p>
            This paper provides an empirical investigation of Kullback-Leibler (KL) divergence through the lens of binary classification using machine learning algorithms and deep learning. We first establish the theoretical foundations of KL divergence and its role in machine learning. Through a series of experiments with three different classification models, we visualize and analyze how the KL divergence evolves during the training process and influences the behavior of the model.
          </p>
        </div>
      )
    },
    methodology: {
      title: 'Methodology',
      icon: FileText,
      content: (
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
      )
    }
  };

  return (
    <div className="website-container">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1>Kullback-Leibler Divergence: A Statistical Bridge</h1>
          <p>Between Information Theory and Machine Learning</p>
        </div>
      </header>

      {/* Main Content */}
      <main className="main-content">
        <div className="content-grid">
          {/* Navigation Sidebar */}
          <nav className="sidebar">
            <div className="nav-container">
              {Object.entries(sections).map(([key, section]) => {
                const Icon = section.icon;
                return (
                  <button
                    key={key}
                    onClick={() => setActiveSection(key as SectionKey)}
                    className={`nav-button ${activeSection === key ? 'active' : ''}`}
                  >
                    <div className="nav-icon">
                      <Icon />
                    </div>
                    {section.title}
                  </button>
                );
              })}
            </div>
          </nav>

          {/* Content Area */}
          <div className="content-area">
            <div className="content-card">
              {sections[activeSection].content}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default ResearchWebsite;