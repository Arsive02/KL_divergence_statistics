import React, { useState } from 'react';
import { Book, Users, FileText, Settings, GitGraph, Layers, Activity, PieChart, Pi, MessageSquare, BookText, BookOpen } from 'lucide-react';
import './index.css';

import Header from './components/Header';
import Navigation from './components/Navigation';
import Introduction from './components/Introduction';
import Authors from './components/Authors';
import Abstract from './components/Abstract';
import TheoreticalFramework from './components/TheoreticalFramework';
import Methodology from './components/Methodology';
import ModelArchitectures from './components/ModelArchitecture';
import TrainingProtocol from './components/TrainingProtocol';
import VisualizationAnalysis from './components/VisualizationAnalysis';
import TrainingDynamics from './components/TrainingDynamics';
import { Section } from './components/types';
import { Conclusion, Discussion, ModelPerformance } from './components/FinalComponents';
import References from './components/References';

const App: React.FC = () => {
  const [activeSection, setActiveSection] = useState('introduction');
  
  const sections: { [key: string]: Section } = {
    authors: {
      title: 'Authors',
      icon: Users,
      component: Authors
    },
    abstract: {
      title: 'Abstract',
      icon: FileText,
      component: Abstract
    },
    introduction: {
      title: 'Introduction',
      icon: Book,
      component: Introduction
    },
    theoreticalFramework: {
      title: 'Theoretical Framework',
      icon: GitGraph,
      component: TheoreticalFramework
    },
    methodology: {
      title: 'Datasets',
      icon: Settings,
      component: Methodology
    },
    modelArchitectures: {
      title: 'Model Architectures',
      icon: Layers,
      component: ModelArchitectures
    },
    trainingProtocol: {
      title: 'Training Protocol',
      icon: Activity,
      component: TrainingProtocol
    },
    visualizationAnalysis: {
      title: 'Visualization Analysis',
      icon: PieChart,
      component: VisualizationAnalysis
    },
    trainingDynamics: {
      title: 'Training Dynamics',
      icon: Pi,
      component: TrainingDynamics
    },
    modelPerformance: {
      title: 'Model Performance',
      icon: Activity,
      component: ModelPerformance
    },
    discussion: {
      title: 'Discussion',
      icon: MessageSquare,
      component: Discussion
    },
    conclusion: {
      title: 'Conclusion',
      icon: BookText,
      component: Conclusion
    },
    references: {
      title: 'References',
      icon: BookOpen,
      component: References
    }
  };

  const ActiveComponent = sections[activeSection].component;

  return (
    <div className="website-container">
      <Header />
      <main className="main-content">
        <div className="content-grid">
          <Navigation 
            activeSection={activeSection}
            setActiveSection={setActiveSection}
            sections={sections}
          />
          <div className="content-area">
            <div className="content-card">
              <ActiveComponent />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;