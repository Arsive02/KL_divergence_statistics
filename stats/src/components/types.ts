import { LucideIcon } from 'lucide-react';

export interface Section {
  title: string;
  icon: LucideIcon;
  component: React.FC;
}

export interface NavigationProps {
  activeSection: string;
  setActiveSection: (section: string) => void;
  sections: {
    [key: string]: Section;
  };
}