import React from 'react';
import { Github, Linkedin } from 'lucide-react';

interface AuthorProps {
  name: string;
  github: string;
  linkedin: string;
}

const authors: AuthorProps[] = [
  {
    name: "Sivakumar Ramakrishnan",
    github: "https://github.com/Arsive02",
    linkedin: "https://www.linkedin.com/in/siva-kumar-5b2527190/"
  },
  {
    name: "Sai Nandini Tata",
    github: "https://github.com/nandinitata",
    linkedin: "https://www.linkedin.com/in/tatasainandini/"
  },
  {
    name: "Deep Shukla",
    github: "https://github.com/deepshukla",
    linkedin: "https://linkedin.com/in/deepshukla"
  }
];

const Authors: React.FC = () => (
  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
    {authors.map((author) => (
      <div key={author.name} className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow duration-300">
        <h3 className="text-xl font-semibold text-slate-800 mb-2">{author.name}</h3>
        <p className="text-slate-600 mb-1">Department of Data Science</p>
        <p className="text-slate-600 mb-4">University of Colorado Boulder</p>
        
        <div className="flex items-center space-x-4 mt-2">
          <a 
            href={author.github}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center text-slate-600 hover:text-slate-900 transition-colors duration-200"
          >
            <Github className="w-5 h-5 mr-1" />
            <span>GitHub</span>
          </a>
          <a 
            href={author.linkedin}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center text-slate-600 hover:text-slate-900 transition-colors duration-200"
          >
            <Linkedin className="w-5 h-5 mr-1" />
            <span>LinkedIn</span>
          </a>
        </div>
      </div>
    ))}
  </div>
);

export default Authors;