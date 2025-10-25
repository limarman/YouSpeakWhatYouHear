import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search } from 'lucide-react';
import type { SearchResult } from '../types';
import { searchMovies } from '../services/api';

export function SearchBar() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const searchRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (searchRef.current && !searchRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    if (query.trim().length < 2) {
      setResults([]);
      setIsOpen(false);
      return;
    }

    const timer = setTimeout(async () => {
      const searchResults = await searchMovies(query);
      setResults(searchResults.slice(0, 5));
      setIsOpen(true);
    }, 300);

    return () => clearTimeout(timer);
  }, [query]);

  function handleSelectMovie(movie: SearchResult) {
    setQuery('');
    setIsOpen(false);
    navigate(`/medium/${movie.id}`);
  }

  return (
    <div ref={searchRef} className="relative w-full max-w-2xl">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-base-content opacity-60 w-5 h-5" />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search movies or series..."
          className="input input-bordered w-full pl-10"
        />
      </div>
      
      {isOpen && results.length > 0 && (
        <div className="absolute top-full mt-2 w-full bg-base-100 border border-base-300 rounded-lg shadow-lg z-50 max-h-96 overflow-y-auto">
          {results.map((movie) => (
            <button
              key={movie.id}
              onClick={() => handleSelectMovie(movie)}
              className="w-full flex items-center gap-3 p-3 hover:bg-base-200 transition-colors text-left"
            >
              <img 
                src={movie.coverUrl} 
                alt={movie.name}
                className="w-12 h-18 object-cover rounded"
              />
              <div>
                <div className="font-medium">{movie.name}</div>
                {movie.year && <div className="text-sm opacity-60">{movie.year}</div>}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}