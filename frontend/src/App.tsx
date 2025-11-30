import { useState, useEffect } from 'react';
import { 
  Search, BookOpen, Loader2, X, Sparkles, LayoutGrid, 
  CheckCircle, Moon, Sun, Info 
} from 'lucide-react';
import { api } from './api';
import { Loader } from './Loader';
import { BookCover } from './BookCover';
import { BookCard } from './BookCard';
import type { RecommendationResult, BookCluster } from './types';

function App() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<RecommendationResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [longLoading, setLongLoading] = useState(false); // New state for slow requests
  const [hasSearched, setHasSearched] = useState(false);
  
  // Modal State
  const [selectedResult, setSelectedResult] = useState<RecommendationResult | null>(null);
  const [explanation, setExplanation] = useState<{ summary: string; details: Record<string, number> } | null>(null);
  const [explaining, setExplaining] = useState(false);
  const [showAbout, setShowAbout] = useState(false);
  
  // Related Books State
  const [relatedBooks, setRelatedBooks] = useState<RecommendationResult[]>([]);
  const [loadingRelated, setLoadingRelated] = useState(false);

  // Dynamic Clusters State
  const [clusters, setClusters] = useState<BookCluster[]>([]);
  const [loadingClusters, setLoadingClusters] = useState(true);

  // Toast State
  const [toast, setToast] = useState<{ message: string; visible: boolean } | null>(null);

  // Settings / Theme State
  const [darkMode, setDarkMode] = useState(false);

  // Load Theme from LocalStorage
  useEffect(() => {
    const savedTheme = localStorage.getItem('bookfinder_theme_mode');
    if (savedTheme) {
        setDarkMode(savedTheme === 'dark');
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        setDarkMode(true);
    }
  }, []);

  // Save Theme to LocalStorage and apply class
  useEffect(() => {
    localStorage.setItem('bookfinder_theme_mode', darkMode ? 'dark' : 'light');
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  // View Transition Toggle
  const toggleTheme = async (e: React.MouseEvent) => {
    const isDark = !darkMode;

    // Fallback for browsers without View Transitions
    if (!(document as any).startViewTransition) {
      setDarkMode(isDark);
      return;
    }

    const x = e.clientX;
    const y = e.clientY;
    const endRadius = Math.hypot(
      Math.max(x, window.innerWidth - x),
      Math.max(y, window.innerHeight - y)
    );

    const transition = (document as any).startViewTransition(() => {
      setDarkMode(isDark);
    });

    await transition.ready;

    const clipPath = [
      `circle(0px at ${x}px ${y}px)`,
      `circle(${endRadius}px at ${x}px ${y}px)`,
    ];

    document.documentElement.animate(
      {
        clipPath: isDark ? clipPath : [...clipPath].reverse(),
      },
      {
        duration: 500,
        easing: 'ease-in-out',
        pseudoElement: isDark ? '::view-transition-new(root)' : '::view-transition-old(root)',
      }
    );
  };

  // Haptic Feedback Helper
  const triggerHaptic = () => {
    if (typeof navigator !== 'undefined' && navigator.vibrate) {
      navigator.vibrate(10);
    }
  };

  useEffect(() => {
    const fetchClusters = async () => {
      try {
        const data = await api.getClusters();
        setClusters(data.slice(0, 6));
      } catch (err) {
        console.error("Failed to fetch clusters", err);
      } finally {
        setLoadingClusters(false);
      }
    };
    fetchClusters();
  }, []);

  const showToast = (message: string) => {
    setToast({ message, visible: true });
    setTimeout(() => setToast(null), 3000);
  };

  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;

    triggerHaptic();
    setLoading(true);
    setLongLoading(false);
    setHasSearched(true);
    setResults([]); 
    
    // Set a timer to show "Waking up" message if it takes too long
    const timer = setTimeout(() => setLongLoading(true), 3000);

    try {
      const data = await api.recommendByQuery(query);
      setResults(data);
    } catch (err) {
      console.error(err);
    } finally {
      clearTimeout(timer);
      setLoading(false);
      setLongLoading(false);
    }
  };

  const handleQuickSearch = (text: string) => {
    setQuery(text);
    // We need to wait for state update or pass text directly. 
    // Simplest is to just call the API logic directly or use a separate effect, 
    // but modifying handleSearch to read state is tricky due to closures.
    // Let's just duplicate the logic slightly or force a search.
    // Actually, setting query then calling a wrapper is better, but let's just do this:
    triggerHaptic();
    setLoading(true);
    setLongLoading(false);
    setHasSearched(true);
    setResults([]);
    const timer = setTimeout(() => setLongLoading(true), 3000);
    
    api.recommendByQuery(text).then(setResults).catch(console.error).finally(() => {
        clearTimeout(timer);
        setLoading(false);
        setLongLoading(false);
    });
  };

  const handleFeedback = async (bookId: string, type: 'positive' | 'negative') => {
      triggerHaptic();
      try {
        await api.submitFeedback(query, bookId, type);
        showToast(type === 'positive' ? "Thanks! We'll show more like this." : "Thanks! We'll tune our results.");
      } catch (err) {
        console.error(err);
      }
  };

  const handleReadMore = async (result: RecommendationResult) => {
    triggerHaptic();
    setSelectedResult(result);
    setExplaining(true);
    setExplanation(null);
    setRelatedBooks([]);
    setLoadingRelated(true);
    
    try {
      const [expl, related] = await Promise.allSettled([
        api.explainRecommendation(query || result.book.title, result.book, result.similarity_score),
        api.getRelatedBooks(result.book.title)
      ]);

      if (expl.status === 'fulfilled') setExplanation(expl.value);
      if (related.status === 'fulfilled') setRelatedBooks(related.value);
      
    } catch (err) {
      console.error("Failed to fetch details", err);
    } finally {
      setExplaining(false);
      setLoadingRelated(false);
    }
  };

  const closeModal = () => {
    setSelectedResult(null);
    setExplanation(null);
    setRelatedBooks([]);
  };

  const handleClusterClick = (clusterName: string) => {
    triggerHaptic();
    setQuery(clusterName);
    setLoading(true);
    setHasSearched(true);
    setResults([]);
    api.recommendByQuery(clusterName).then(setResults).catch(console.error).finally(() => setLoading(false));
  };

  return (
    <div className={`min-h-screen font-sans selection:bg-indigo-500 selection:text-white transition-colors duration-300 ${darkMode ? 'bg-zinc-950 text-zinc-100' : 'bg-zinc-50 text-zinc-900'}`}>
      
      {/* Dynamic Background */}
      <div className="fixed inset-0 -z-10 bg-zinc-50 dark:bg-zinc-950 transition-colors duration-500" />
      <div className="fixed inset-0 -z-10 bg-noise opacity-[0.04] dark:opacity-[0.06] pointer-events-none mix-blend-overlay" />
      <div className="fixed inset-0 -z-10 bg-dot-pattern pointer-events-none" />
      <div className="fixed inset-0 -z-10 bg-gradient-to-br from-rose-100/40 via-white to-indigo-100/40 dark:from-indigo-950/30 dark:via-zinc-950 dark:to-purple-950/30 pointer-events-none" />


      {/* Toast */}
      {toast && (
        <div className="fixed bottom-6 right-6 bg-zinc-900 dark:bg-zinc-800 text-white px-4 py-3 rounded-xl shadow-2xl flex items-center gap-3 animate-slide-up z-[60] border border-zinc-800 dark:border-zinc-700">
          <CheckCircle className="w-5 h-5 text-green-400" />
          <span className="text-sm font-medium">{toast.message}</span>
        </div>
      )}

      {/* Header */}
      <header className="sticky top-0 z-30 border-b border-zinc-200/50 dark:border-zinc-800/50 bg-white/70 dark:bg-zinc-950/70 backdrop-blur-xl">
        <div className="max-w-3xl mx-auto px-6 h-16 flex items-center justify-between">
          <div 
            className="flex items-center gap-2 font-bold tracking-tight text-lg cursor-pointer hover:opacity-70 transition-opacity" 
            onClick={() => { triggerHaptic(); setHasSearched(false); setResults([]); setQuery(''); }}
          >
            <div className="bg-indigo-600 text-white p-1.5 rounded-lg shadow-lg shadow-indigo-500/20">
                <BookOpen strokeWidth={3} className="w-4 h-4" />
            </div>
            <span className="text-zinc-900 dark:text-white font-serif text-xl tracking-normal">BookFinder</span>
          </div>
          
          <div className="flex items-center gap-2">
             <button 
                onClick={toggleTheme}
                className="p-2 rounded-full hover:bg-black/5 dark:hover:bg-white/10 transition-colors text-zinc-500 dark:text-zinc-400"
                aria-label="Toggle Dark Mode"
             >
                {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
             </button>
             <button 
                onClick={() => setShowAbout(true)} 
                className="p-2 ml-2 rounded-full hover:bg-black/5 dark:hover:bg-white/10 transition-colors text-zinc-500 dark:text-zinc-400"
                aria-label="About BookFinder"
             >
                <Info className="w-5 h-5" />
             </button>
          </div>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-6 pb-24 relative z-10">
        {/* Search Section */}
        <div className={`transition-all duration-700 ease-out ${hasSearched ? 'py-10' : 'py-28'}`}>
          {!hasSearched && (
            <div className="text-center mb-10 animate-fade-in">
              <h1 className="text-5xl sm:text-7xl font-black tracking-tighter mb-6 text-zinc-900 dark:text-white font-serif">
                Find your next<br/>
                <span className="text-indigo-600 dark:text-indigo-500 italic">obsession.</span>
              </h1>
              <p className="text-lg max-w-md mx-auto text-zinc-500 dark:text-zinc-400 leading-relaxed font-medium">
                Describe the vibe, plot, or character you're looking for.
              </p>
            </div>
          )}
          
          <form onSubmit={handleSearch} className="relative max-w-xl mx-auto z-10">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. A sci-fi thriller about AI consciousness..."
              className="relative w-full bg-white dark:bg-zinc-900 border-2 border-zinc-200 dark:border-zinc-800 focus:border-indigo-500 dark:focus:border-indigo-500 rounded-full py-4 pl-6 pr-24 text-lg outline-none transition-all placeholder:text-zinc-400 dark:placeholder:text-zinc-600 shadow-xl shadow-zinc-200/50 dark:shadow-black/50 text-zinc-900 dark:text-white"
              autoFocus
            />
            
            {/* Clear Button */}
            {query && (
              <button
                type="button"
                onClick={() => {
                  setQuery('');
                  setResults([]);
                  setHasSearched(false);
                  triggerHaptic();
                }}
                className="absolute right-14 top-1/2 -translate-y-1/2 p-2 text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 transition-colors"
                aria-label="Clear search"
              >
                <X className="w-5 h-5" />
              </button>
            )}

            <button 
              type="submit"
              disabled={loading || !query.trim()}
              className="absolute right-2 top-2 p-2.5 bg-indigo-600 text-white rounded-full hover:bg-indigo-700 disabled:opacity-50 disabled:hover:bg-indigo-600 transition-all active:scale-90 shadow-lg shadow-indigo-500/30"
              onClick={triggerHaptic}
            >
              {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Search className="w-5 h-5" />}
            </button>
          </form>
          
          {/* Long Loading Message */}
          {loading && longLoading && (
            <div className="text-center mt-4 animate-fade-in">
              <p className="text-sm font-medium text-indigo-600 dark:text-indigo-400 flex items-center justify-center gap-2">
                <Loader2 className="w-3 h-3 animate-spin" />
                Waking up the AI... this might take a moment 😴
              </p>
            </div>
          )}

          {/* Quick Prompts */}
          {!hasSearched && !loading && (
            <div className="mt-8 flex flex-wrap justify-center gap-3 animate-fade-in animation-delay-200">
               {[
                 "Cyberpunk noir detective",
                 "Cozy cottagecore mystery",
                 "Space opera with politics",
                 "Psychological horror 1920s"
               ].map((prompt) => (
                 <button
                   key={prompt}
                   onClick={() => handleQuickSearch(prompt)}
                   className="px-4 py-2 rounded-full text-sm font-medium bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800 text-zinc-600 dark:text-zinc-400 hover:border-indigo-400 dark:hover:border-indigo-500 hover:text-indigo-600 dark:hover:text-indigo-400 transition-all shadow-sm hover:shadow-md active:scale-95"
                 >
                   ✨ {prompt}
                 </button>
               ))}
            </div>
          )}

          {!hasSearched && (
            <div className="mt-20 animate-slide-up animation-delay-500">
              <div className="flex items-center justify-center gap-2 text-xs uppercase tracking-widest font-bold mb-8 text-zinc-400 dark:text-zinc-600">
                 <LayoutGrid className="w-4 h-4" />
                 Curated Collections
              </div>
              
              {loadingClusters ? (
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                   {[1,2,3,4,5,6].map(i => (
                     <div key={i} className="h-32 bg-white dark:bg-zinc-900 rounded-2xl border border-zinc-200 dark:border-zinc-800 animate-pulse" />
                   ))}
                </div>
              ) : (
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  {clusters.map(cluster => (
                    <button 
                      key={cluster.id}
                      onClick={() => handleClusterClick(cluster.name)}
                      className="text-left p-4 bg-white dark:bg-zinc-900/50 hover:bg-zinc-50 dark:hover:bg-zinc-800 border border-zinc-200 dark:border-zinc-800 hover:border-indigo-300 dark:hover:border-indigo-500/50 transition-all rounded-2xl group active:scale-95 duration-200 shadow-sm hover:shadow-md"
                    >
                      <h3 className="font-bold text-zinc-800 dark:text-zinc-200 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 truncate transition-colors">{cluster.name}</h3>
                      <p className="text-xs text-zinc-500 dark:text-zinc-500 mt-1 font-medium">{cluster.size} books</p>
                      
                      <div className="flex mt-4 -space-x-3 overflow-hidden px-1 pb-1">
                        {cluster.top_books.slice(0, 3).map((book, i) => (
                          <div key={book.id} className={`w-8 h-12 bg-zinc-200 dark:bg-zinc-800 rounded-sm shadow-md border border-white/50 dark:border-zinc-700 relative z-${3-i} transform transition-transform group-hover:-translate-y-1 duration-300 overflow-hidden`} style={{transitionDelay: `${i*50}ms`}}>
                             <BookCover src={book.cover_image_url} title={book.title} author={book.authors[0]} className="w-full h-full" />
                          </div>
                        ))}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Results / Skeletons */}
        <div className="space-y-6">
          {loading && results.length === 0 ? (
             <Loader />
          ) : (
            results.map((result) => (
              <BookCard 
                key={result.book.id} 
                result={result} 
                onClick={() => handleReadMore(result)} 
                onFeedback={handleFeedback} 
              />
            ))
          )}
        </div>

        {hasSearched && results.length === 0 && !loading && (
          <div className="text-center text-zinc-400 dark:text-zinc-600 py-12">
            No books found matching that description.
          </div>
        )}
      </main>

      {/* Detail Modal */}
      {selectedResult && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6">
          <div className="absolute inset-0 bg-zinc-900/60 dark:bg-black/80 backdrop-blur-sm transition-opacity" onClick={closeModal} />
          <div className="bg-white dark:bg-zinc-900 rounded-3xl shadow-2xl w-full max-w-2xl max-h-[90vh] overflow-y-auto relative animate-slide-up z-50 no-scrollbar border border-zinc-200 dark:border-zinc-800">
            <button 
                onClick={() => { triggerHaptic(); closeModal(); }} 
                className="absolute right-4 top-4 p-2 bg-white/80 dark:bg-black/50 backdrop-blur hover:bg-zinc-100 dark:hover:bg-zinc-800 text-zinc-500 dark:text-zinc-400 rounded-full transition-colors z-10 shadow-sm"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="grid sm:grid-cols-[220px_1fr] gap-0 sm:gap-8">
               <div className="bg-zinc-100 dark:bg-zinc-800 h-64 sm:h-auto sm:aspect-[2/3] relative overflow-hidden">
                  <BookCover 
                    src={selectedResult.book.cover_image_url} 
                    title={selectedResult.book.title} 
                    author={selectedResult.book.authors[0]}
                    className="w-full h-full"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent sm:hidden"></div>
                  <h2 className="absolute bottom-4 left-4 right-4 text-2xl font-bold font-serif text-white sm:hidden leading-tight shadow-black drop-shadow-md">
                    {selectedResult.book.title}
                  </h2>
               </div>
               
               <div className="p-6 sm:pl-0 sm:py-8 space-y-6">
                  <div className="hidden sm:block">
                    <h2 className="text-3xl sm:text-4xl font-bold font-serif leading-tight mb-2 text-zinc-900 dark:text-white">{selectedResult.book.title}</h2>
                    <p className="text-lg text-zinc-500 dark:text-zinc-400 font-medium">by {selectedResult.book.authors.join(', ')}</p>
                  </div>

                  <div className="prose prose-zinc dark:prose-invert prose-sm max-w-none leading-relaxed">
                    <p>{selectedResult.book.description}</p>
                  </div>

                  {/* AI Explanation Section */}
                  <div className="bg-indigo-50/50 dark:bg-indigo-900/20 rounded-2xl p-5 border border-indigo-100 dark:border-indigo-500/20">
                    <div className="flex items-center gap-2 mb-3 text-indigo-600 dark:text-indigo-400 font-bold text-xs uppercase tracking-widest">
                      <Sparkles className="w-4 h-4" />
                      Why this match?
                    </div>
                    
                    {explaining ? (
                      <div className="flex items-center gap-2 text-indigo-400 dark:text-indigo-300 text-sm font-medium">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Reading your mind...
                      </div>
                    ) : explanation ? (
                      <div className="space-y-3">
                        <p className="text-sm text-zinc-800 dark:text-zinc-200 font-medium leading-relaxed">
                          {explanation.summary}
                        </p>
                        <div className="flex flex-wrap gap-2">
                          {Object.entries(explanation.details).map(([key, val]) => (
                            <span key={key} className="text-[10px] font-bold uppercase bg-white dark:bg-zinc-800 border border-indigo-100 dark:border-indigo-500/30 px-2 py-1 rounded-md text-indigo-600 dark:text-indigo-300 shadow-sm">
                              {key} {val}%
                            </span>
                          ))}
                        </div>
                      </div>
                    ) : (
                       <p className="text-sm text-red-400">Could not generate explanation.</p>
                    )}
                  </div>

                  <div className="pt-4 flex flex-col sm:flex-row gap-3">
                    <a 
                      href={`https://www.goodreads.com/search?q=${encodeURIComponent(selectedResult.book.title + ' ' + selectedResult.book.authors.join(', '))}`} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-2 px-6 py-3 bg-white dark:bg-zinc-800 text-zinc-900 dark:text-white border border-zinc-200 dark:border-zinc-700 font-bold rounded-full hover:bg-zinc-50 dark:hover:bg-zinc-700 hover:scale-105 transition-all active:scale-95 flex-1 justify-center shadow-sm"
                      onClick={triggerHaptic}
                    >
                      <BookOpen className="w-4 h-4" />
                      Goodreads
                    </a>
                    
                    <a 
                      href={`https://www.google.com/search?q=${encodeURIComponent(selectedResult.book.title + ' by ' + selectedResult.book.authors.join(', '))}`} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-2 px-6 py-3 bg-zinc-900 dark:bg-indigo-600 text-white font-bold rounded-full hover:bg-black dark:hover:bg-indigo-700 hover:scale-105 transition-all active:scale-95 flex-1 justify-center shadow-xl shadow-zinc-900/20 dark:shadow-indigo-500/30"
                      onClick={triggerHaptic}
                    >
                      <Search className="w-4 h-4" />
                      Google
                    </a>
                  </div>

                  {/* Related Books Section */}
                  <div className="pt-8 border-t border-dashed border-zinc-200 dark:border-zinc-800">
                    <h3 className="text-xs font-bold text-zinc-400 dark:text-zinc-500 mb-4 uppercase tracking-widest">You might also like</h3>
                    {loadingRelated ? (
                        <div className="flex gap-4">
                            {[1, 2, 3].map(i => (
                                <div key={i} className="w-32 h-48 bg-zinc-100 dark:bg-zinc-800 rounded-xl animate-pulse" />
                            ))}
                        </div>
                    ) : (
                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                            {relatedBooks.map(rel => (
                                <div 
                                    key={rel.book.id} 
                                    onClick={() => handleReadMore(rel)}
                                    className="group cursor-pointer active:scale-95 transition-transform"
                                >
                                    <div className="aspect-[2/3] bg-zinc-100 dark:bg-zinc-800 rounded-xl overflow-hidden mb-2 relative shadow-sm group-hover:shadow-md transition-all border border-zinc-100 dark:border-zinc-700">
                                        <BookCover 
                                          src={rel.book.cover_image_url} 
                                          title={rel.book.title} 
                                          author={rel.book.authors[0]} 
                                          className="w-full h-full group-hover:scale-105 transition-transform duration-500"
                                        />
                                    </div>
                                    <h4 className="text-xs font-bold leading-tight text-zinc-800 dark:text-zinc-200 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors line-clamp-1">{rel.book.title}</h4>
                                    <p className="text-[10px] text-zinc-400 dark:text-zinc-500 mt-0.5 truncate">{rel.book.authors[0]}</p>
                                </div>
                            ))}
                        </div>
                    )}
                  </div>

               </div>
            </div>
          </div>
        </div>
      )}

      {/* About Modal */}
      {showAbout && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div className="absolute inset-0 bg-zinc-900/60 dark:bg-black/80 backdrop-blur-sm transition-opacity" onClick={() => setShowAbout(false)} />
          <div className="bg-white dark:bg-zinc-900 rounded-3xl shadow-2xl w-full max-w-lg max-h-[90vh] overflow-y-auto relative animate-slide-up z-50 no-scrollbar border border-zinc-200 dark:border-zinc-800">
            <button 
                onClick={() => { triggerHaptic(); setShowAbout(false); }} 
                className="absolute right-4 top-4 p-2 bg-white/80 dark:bg-black/50 backdrop-blur hover:bg-zinc-100 dark:hover:bg-zinc-800 text-zinc-500 dark:text-zinc-400 rounded-full transition-colors z-10 shadow-sm"
            >
              <X className="w-5 h-5" />
            </button>
            <div className="p-6 sm:p-8 space-y-6">
              <h3 className="text-2xl font-bold text-zinc-900 dark:text-white">About BookFinder AI</h3>
              <p className="text-zinc-600 dark:text-zinc-300 leading-relaxed">
                BookFinder AI is an intelligent book recommendation engine powered by cutting-edge Generative AI. 
                It leverages advanced semantic search to understand your queries and recommend books based on vibe, plot, characters, and themes, rather than just keywords.
              </p>
              <div className="space-y-3">
                <h4 className="text-lg font-bold text-zinc-800 dark:text-zinc-200">How it works:</h4>
                <ul className="list-disc list-inside text-zinc-600 dark:text-zinc-300 space-y-1">
                  <li>**Semantic Search:** Understands the meaning and context of your queries.</li>
                  <li>**Large Language Models (LLM):** Used for generating personalized explanations.</li>
                  <li>**Vector Databases:** Efficiently stores and retrieves book embeddings for fast recommendations.</li>
                  <li>**Machine Learning:** Constantly improves recommendations based on user feedback.</li>
                </ul>
              </div>
              <p className="text-zinc-600 dark:text-zinc-300 leading-relaxed">
                Describe your ideal read and let BookFinder AI uncover your next literary obsession!
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;