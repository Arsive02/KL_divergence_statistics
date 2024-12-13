import React from 'react';
import { Github, Linkedin, Mail, Building2, MapPin } from 'lucide-react';

interface AuthorProps {
  name: string;
  github: string;
  linkedin: string;
  email: string;
}

const authors: AuthorProps[] = [
  {
    name: "Sivakumar Ramakrishnan",
    github: "https://github.com/Arsive02",
    linkedin: "https://www.linkedin.com/in/siva-kumar-5b2527190/",
    email: "sivakumar.ramakrishnan@colorado.edu"
  },
  {
    name: "Sai Nandini Tata",
    github: "https://github.com/nandinitata",
    linkedin: "https://www.linkedin.com/in/tatasainandini/",
    email: "saiNandini.Tata@colorado.edu"
  },
  {
    name: "Deep Shukla",
    github: "https://github.com/623dks",
    linkedin: "https://www.linkedin.com/in/deep-shukla-b4035220a/",
    email: "Deep.Shukla@colorado.edu"
  }
];

const Authors: React.FC = () => (
  <div className="max-w-4xl mx-auto p-6">
    <h2 className="text-3xl font-serif text-slate-800 mb-8">Authors</h2>
    
    <div className="space-y-4">
      {authors.map((author) => (
        <div 
          key={author.name} 
          className="bg-white rounded-lg border border-slate-200 overflow-hidden transition-all duration-300 hover:shadow-lg"
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-6">
            {/* Author Info */}
            <div className="flex flex-col justify-center">
              <h3 className="text-xl font-semibold text-slate-800 mb-3">{author.name}</h3>
              <div className="space-y-2">
                <div className="flex items-center text-slate-600">
                  <Building2 className="w-4 h-4 mr-2 flex-shrink-0" />
                  <span>Department of Data Science</span>
                </div>
                <div className="flex items-center text-slate-600">
                  <MapPin className="w-4 h-4 mr-2 flex-shrink-0" />
                  <span>University of Colorado Boulder</span>
                </div>
              </div>
            </div>

            {/* Contact Links */}
            <div className="flex flex-col justify-center space-y-2">
              <a 
                href={author.github}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center px-4 py-2 rounded-md bg-slate-50 text-slate-700 hover:bg-slate-100 transition-all duration-200 hover:translate-x-1"
              >
                <Github className="w-5 h-5 mr-3" />
                <span>GitHub Profile</span>
              </a>
              
              <a 
                href={author.linkedin}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center px-4 py-2 rounded-md bg-slate-50 text-slate-700 hover:bg-slate-100 transition-all duration-200 hover:translate-x-1"
              >
                <Linkedin className="w-5 h-5 mr-3" />
                <span>LinkedIn Profile</span>
              </a>

              <a 
                href={`mailto:${author.email}`}
                className="inline-flex items-center px-4 py-2 rounded-md bg-slate-50 text-slate-700 hover:bg-slate-100 transition-all duration-200 hover:translate-x-1"
              >
                <Mail className="w-5 h-5 mr-3" />
                <span>Contact via Email</span>
              </a>
            </div>
          </div>
        </div>
      ))}
    </div>

    <div className="mt-8 text-center text-slate-600">
      <p className="text-sm">
        Research conducted at the Department of Data Science,
        <br />
        University of Colorado Boulder
      </p>
    </div>
  </div>
);

export default Authors;