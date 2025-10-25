export interface SearchResult {
    id: string;
    name: string;
    coverUrl: string;
    year?: number;
    type: 'movie' | 'series';
  }
  
  export interface MovieData {
    totalSpeechTime: number; // in minutes
  }