import React, { useState } from 'react';
import { Book, Users, FileText, Settings } from 'lucide-react';
import './index.css';

import Header from './components/Header';
import Navigation from './components/Navigation';
import Introduction from './components/Introduction';
import Authors from './components/Authors';
import Abstract from './components/Abstract';
import Methodology from './components/Methodology';
import { Section } from './components/types';

const App: React.FC = () => {
  const [activeSection, setActiveSection] = useState('introduction');
  
  const sections: { [key: string]: Section } = {
    introduction: {
      title: 'Introduction',
      icon: Book,
      component: Introduction
    },
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
    methodology: {
      title: 'Methodology',
      icon: Settings,
      component: Methodology
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