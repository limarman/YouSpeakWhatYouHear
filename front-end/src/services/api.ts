import type { SearchResult, MovieData } from '../types';

// Mock data for development
const mockSearchResults: SearchResult[] = [
  { id: '1', name: 'The Shawshank Redemption', coverUrl: 'https://via.placeholder.com/300x450/4A5568/FFFFFF?text=Shawshank', year: 1994, type: 'movie' },
  { id: '2', name: 'The Godfather', coverUrl: 'https://via.placeholder.com/300x450/4A5568/FFFFFF?text=Godfather', year: 1972, type: 'movie' },
  { id: '3', name: 'The Dark Knight', coverUrl: 'https://via.placeholder.com/300x450/4A5568/FFFFFF?text=Dark+Knight', year: 2008, type: 'movie' },
  { id: '4', name: 'Pulp Fiction', coverUrl: 'https://via.placeholder.com/300x450/4A5568/FFFFFF?text=Pulp+Fiction', year: 1994, type: 'movie' },
  { id: '5', name: 'Forrest Gump', coverUrl: 'https://via.placeholder.com/300x450/4A5568/FFFFFF?text=Forrest+Gump', year: 1994, type: 'movie' },
];

export async function searchMovies(query: string): Promise<SearchResult[]> {
  // TODO: Replace with actual API call
  // const response = await fetch(`/search?query=${encodeURIComponent(query)}`);
  // return await response.json();
  
  await new Promise(resolve => setTimeout(resolve, 300));
  return mockSearchResults.filter(m => 
    m.name.toLowerCase().includes(query.toLowerCase())
  );
}

export async function getMovieData(id: string): Promise<MovieData> {
  // TODO: Replace with actual API call
  // const response = await fetch(`/data?id=${encodeURIComponent(id)}`);
  // return await response.json();
  
  await new Promise(resolve => setTimeout(resolve, 300));
  return { totalSpeechTime: Math.floor(Math.random() * 120) + 30 };
}

// Helper to get movie info by ID (for direct navigation)
export async function getMovieById(id: string): Promise<SearchResult | null> {
  // TODO: You might want a dedicated endpoint for this
  // const response = await fetch(`/movie?id=${encodeURIComponent(id)}`);
  // return await response.json();
  
  await new Promise(resolve => setTimeout(resolve, 100));
  return mockSearchResults.find(m => m.id === id) || null;
}
