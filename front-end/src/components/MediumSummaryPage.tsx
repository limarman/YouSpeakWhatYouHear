import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Clock, ArrowLeft } from 'lucide-react';
import type { SearchResult, MovieData } from '../types';
import { getMovieById, getMovieData } from '../services/api';

export function MediumSummaryPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [movie, setMovie] = useState<SearchResult | null>(null);
  const [movieData, setMovieData] = useState<MovieData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    async function loadData() {
      if (!id) {
        setError(true);
        setLoading(false);
        return;
      }

      setLoading(true);
      setError(false);

      try {
        const movieInfo = await getMovieById(id);
        if (!movieInfo) {
          setError(true);
          setLoading(false);
          return;
        }

        setMovie(movieInfo);
        const data = await getMovieData(id);
        setMovieData(data);
      } catch (err) {
        setError(true);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, [id]);

  function formatTime(minutes: number): string {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-base-200 flex items-center justify-center">
        <span className="loading loading-spinner loading-lg"></span>
      </div>
    );
  }

  if (error || !movie) {
    return (
      <div className="min-h-screen bg-base-200 flex items-center justify-center p-4">
        <div className="text-center">
          <h2 className="text-3xl font-bold mb-2">Movie not found</h2>
          <p className="mb-6 opacity-80">The movie you're looking for doesn't exist.</p>
          <button
            onClick={() => navigate('/')}
            className="btn btn-primary"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to home
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-base-200">
      <div className="max-w-4xl mx-auto p-4 pb-20">
        <button 
          onClick={() => navigate('/')}
          className="btn btn-ghost btn-sm mb-4"
        >
          <ArrowLeft className="w-4 h-4" />
          Back
        </button>
        
        <div className="card bg-base-100 shadow-xl">
          <div className="card-body p-0">
            <div className="md:flex">
              <div className="md:w-1/3">
                <img 
                  src={movie.coverUrl} 
                  alt={movie.name}
                  className="w-full h-auto object-cover rounded-t-2xl md:rounded-l-2xl md:rounded-tr-none"
                />
              </div>
              
              <div className="md:w-2/3 p-6">
                <h1 className="card-title text-3xl mb-2">{movie.name}</h1>
                {movie.year && <p className="opacity-70 mb-6">{movie.year}</p>}
                
                <div className="divider"></div>
                
                <h2 className="text-xl font-semibold mb-4">Speech Analytics</h2>
                
                {movieData ? (
                  <div className="stats shadow">
                    <div className="stat">
                      <div className="stat-figure text-primary">
                        <Clock className="w-8 h-8" />
                      </div>
                      <div className="stat-title">Total Speech Time</div>
                      <div className="stat-value text-primary">
                        {formatTime(movieData.totalSpeechTime)}
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="alert">
                    <span>No data available</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}