import React from 'react';
import { NavigationProps } from './types';

const Navigation: React.FC<NavigationProps> = ({ activeSection, setActiveSection, sections }) => (
  <nav className="sidebar">
    <div className="nav-container">
      {Object.entries(sections).map(([key, section]) => {
        const Icon = section.icon;
        return (
          <button
            key={key}
            onClick={() => setActiveSection(key)}
            className={`nav-button ${activeSection === key ? 'active' : ''}`}
          >
            <Icon className="nav-icon" />
            {section.title}
          </button>
        );
      })}
    </div>
  </nav>
);

export default Navigation;