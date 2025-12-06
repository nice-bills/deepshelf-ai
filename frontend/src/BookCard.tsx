import React from 'react';
import { ThumbsUp, ThumbsDown, ArrowRight, Sparkles, Bookmark, Check } from 'lucide-react';
import { BookCover } from './BookCover';
import type { RecommendationResult } from './types';

interface BookCardProps {
  result: RecommendationResult;
  isRead?: boolean;
  onToggleRead?: (title: string) => void;
  onClick: () => void;
  onFeedback: (id: string, type: 'positive' | 'negative') => void;
}

export function BookCard({ result, isRead, onToggleRead, onClick, onFeedback }: BookCardProps) {
  const { book, similarity_score } = result;
  const percentage = Math.round(similarity_score * 100);

  const handleToggleRead = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onToggleRead) {
      onToggleRead(book.title);
    }
  };

  return (
    <div 
      onClick={onClick}
      className={`group relative bg-white dark:bg-zinc-900/80 backdrop-blur-sm border rounded-3xl p-4 sm:p-5 shadow-sm hover:shadow-xl hover:shadow-indigo-500/5 dark:hover:shadow-indigo-500/10 transition-all duration-300 cursor-pointer hover:-translate-y-0.5 active:scale-[0.99] animate-slide-up h-full flex flex-col sm:flex-row gap-5 overflow-hidden ${isRead ? 'border-indigo-500/50 dark:border-indigo-500/50 ring-1 ring-indigo-500/20' : 'border-zinc-200 dark:border-zinc-800'}`}
    >
      {/* Read Status Toggle (Absolute Top Right) */}
      {onToggleRead && (
        <button
          onClick={handleToggleRead}
          className={`absolute top-3 right-3 z-20 p-2 rounded-full transition-all shadow-sm ${isRead ? 'bg-indigo-600 text-white' : 'bg-white/80 dark:bg-zinc-800/80 text-zinc-400 hover:text-indigo-600 dark:hover:text-indigo-400 border border-zinc-200 dark:border-zinc-700'}`}
          title={isRead ? "Remove from history" : "Mark as read"}
        >
          {isRead ? <Check className="w-4 h-4" /> : <Bookmark className="w-4 h-4" />}
        </button>
      )}

      {/* Cover Image Section */}
      <div className="w-full sm:w-28 aspect-[2/3] bg-zinc-100 dark:bg-zinc-800 rounded-xl overflow-hidden relative shadow-inner shrink-0 border border-zinc-100 dark:border-zinc-700">
        <BookCover 
          src={book.cover_image_url} 
          title={book.title} 
          author={book.authors?.[0] || 'Unknown Author'}
          className="w-full h-full transition-transform duration-500 group-hover:scale-105"
        />
        
        {/* Match Badge (Mobile Overlay) */}
        <div className="absolute top-2 right-2 sm:hidden">
          <span className="bg-white/90 dark:bg-black/80 backdrop-blur text-indigo-600 dark:text-indigo-400 text-[10px] font-bold px-2 py-1 rounded-full border border-black/5 dark:border-white/10 shadow-sm">
            {percentage}% Match
          </span>
        </div>
      </div>

      {/* Content Section */}
      <div className="space-y-2.5 flex-1 min-w-0 flex flex-col">
        <div className="flex justify-between items-start gap-4">
          <h2 className="text-xl font-bold leading-tight text-zinc-900 dark:text-zinc-100 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors line-clamp-2 font-serif tracking-tight">
            {book.title}
          </h2>
          {/* Desktop Match Badge */}
          <span className="hidden sm:inline-flex text-xs font-bold text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-500/10 border border-indigo-100 dark:border-indigo-500/20 px-2 py-1 rounded-full whitespace-nowrap items-center gap-1">
            <Sparkles className="w-3 h-3" />
            {percentage}%
          </span>
        </div>
        
        <p className="text-zinc-500 dark:text-zinc-400 font-medium text-xs">
          by <span className="text-zinc-900 dark:text-zinc-200">{book.authors?.join(', ') || 'Unknown Author'}</span>
        </p>

        <p className="text-zinc-600 dark:text-zinc-400 leading-relaxed text-sm line-clamp-2 sm:line-clamp-3">
          {book.description || 'No description available.'}
        </p>

        <div className="pt-2 mt-auto flex items-center justify-between gap-4">
          {/* Mobile: Simple 'Read more' */}
          <button className="flex items-center gap-1 text-sm font-bold text-indigo-600 dark:text-indigo-400 group-active:opacity-70">
            Read more <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
          </button>
          
          {/* Feedback Actions (Desktop Hover / Mobile Always Visible but subtle) */}
          <div 
            className="flex gap-1 sm:opacity-0 sm:group-hover:opacity-100 transition-opacity" 
            onClick={(e) => e.stopPropagation()}
          >
            <button 
              onClick={() => onFeedback(book.id, 'positive')} 
              className="p-2 hover:bg-green-50 dark:hover:bg-green-900/30 text-zinc-400 hover:text-green-600 dark:hover:text-green-400 rounded-full transition-colors"
              aria-label="Like recommendation"
            >
              <ThumbsUp className="w-4 h-4" />
            </button>
            <button 
              onClick={() => onFeedback(book.id, 'negative')} 
              className="p-2 hover:bg-red-50 dark:hover:bg-red-900/30 text-zinc-400 hover:text-red-600 dark:hover:text-red-400 rounded-full transition-colors"
              aria-label="Dislike recommendation"
            >
              <ThumbsDown className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Genre Tags */}
        <div className="flex flex-wrap gap-1.5 pt-2">
          {book.genres?.slice(0, 3).map(genre => (
            <span 
              key={genre} 
              className="text-[10px] uppercase tracking-wider font-bold text-zinc-500 dark:text-zinc-500 border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800 px-2 py-1 rounded-md truncate max-w-[100px]" 
              title={genre}
            >
              {genre}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
