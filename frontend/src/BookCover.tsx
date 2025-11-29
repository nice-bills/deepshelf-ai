import { useState } from 'react';
import { BookOpen } from 'lucide-react';

// Deterministic color generator based on string
const getColor = (str: string) => {
  const colors = [
    'from-red-500 to-orange-500',
    'from-orange-500 to-amber-500',
    'from-amber-500 to-yellow-500',
    'from-yellow-500 to-lime-500',
    'from-lime-500 to-green-500',
    'from-green-500 to-emerald-500',
    'from-emerald-500 to-teal-500',
    'from-teal-500 to-cyan-500',
    'from-cyan-500 to-sky-500',
    'from-sky-500 to-blue-500',
    'from-blue-500 to-indigo-500',
    'from-indigo-500 to-violet-500',
    'from-violet-500 to-purple-500',
    'from-purple-500 to-fuchsia-500',
    'from-fuchsia-500 to-pink-500',
    'from-pink-500 to-rose-500',
    'from-slate-500 to-zinc-500',
  ];
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  return colors[Math.abs(hash) % colors.length];
};

interface BookCoverProps {
  src?: string | null;
  title: string;
  author?: string;
  className?: string;
}

export function BookCover({ src, title, author, className = "" }: BookCoverProps) {
  const [error, setError] = useState(false);
  const gradient = getColor(title);

  if (src && !error) {
    return (
      <img 
        src={src} 
        alt={title}
        onError={() => setError(true)}
        className={`object-cover ${className}`} 
      />
    );
  }

  // Fallback: Generated Cover
  return (
    <div className={`bg-gradient-to-br ${gradient} p-4 flex flex-col justify-between relative overflow-hidden ${className}`}>
      {/* Texture Overlay */}
      <div className="absolute inset-0 bg-noise opacity-20 mix-blend-overlay"></div>
      
      <div className="relative z-10">
        <h3 className="text-white font-bold leading-tight text-shadow-sm line-clamp-4" style={{ fontSize: 'clamp(0.75rem, 1vw, 1rem)' }}>
          {title}
        </h3>
      </div>
      
      {author && (
         <p className="relative z-10 text-white/90 text-[10px] font-medium truncate mt-2">
            {author}
         </p>
      )}

      <BookOpen className="absolute bottom-[-10%] right-[-10%] w-[60%] h-[60%] text-white/10 -rotate-12" />
    </div>
  );
}
