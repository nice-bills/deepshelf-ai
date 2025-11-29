import { BookOpen } from 'lucide-react';
import { useState, useEffect } from 'react';

const LOADING_MESSAGES = [
  "Scanning literary universe...",
  "Analyzing plot vectors...",
  "Connecting thematic dots...",
  "Filtering hidden gems...",
  "Synthesizing recommendations..."
];

export function Loader() {
  const [messageIndex, setMessageIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % LOADING_MESSAGES.length);
    }, 800);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex flex-col items-center justify-center py-24 relative overflow-hidden">
      
      {/* Orbiting / Pulsing Effect */}
      <div className="relative flex items-center justify-center">
        {/* Core */}
        <div className="relative z-10 bg-white dark:bg-zinc-800 p-4 rounded-2xl shadow-xl border border-indigo-100 dark:border-indigo-500/30 animate-bounce-slight">
          <BookOpen className="w-8 h-8 text-indigo-600 dark:text-indigo-400" />
        </div>
        
        {/* Orbital Ring 1 */}
        <div className="absolute w-24 h-24 border-2 border-indigo-500/20 dark:border-indigo-400/20 rounded-full animate-spin-slow" style={{ borderRadius: '40% 60% 70% 30% / 40% 50% 60% 50%' }}></div>
        
        {/* Orbital Ring 2 */}
        <div className="absolute w-32 h-32 border border-purple-500/20 dark:border-purple-400/20 rounded-full animate-spin-reverse-slower" style={{ borderRadius: '60% 40% 30% 70% / 60% 30% 70% 40%' }}></div>
        
        {/* Pulsing Aura */}
        <div className="absolute inset-0 bg-indigo-500/10 dark:bg-indigo-400/10 rounded-full animate-ping-slow"></div>
      </div>

      {/* Text */}
      <div className="mt-8 text-center z-10">
        <p className="text-sm font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600 dark:from-indigo-400 dark:to-purple-400 animate-pulse">
          {LOADING_MESSAGES[messageIndex]}
        </p>
      </div>
    </div>
  );
}
