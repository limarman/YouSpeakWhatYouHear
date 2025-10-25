import React from 'react';
import { SearchBar } from './SearchBar';

export function LandingPage() {
  return (
    <div className="hero min-h-screen bg-base-200">
      <div className="hero-content text-center">
        <div className="max-w-2xl">
          <h1 className="text-5xl font-bold mb-4">Movie Browser</h1>
          <p className="mb-8 text-lg opacity-80">
            Search for movies and discover speech time analytics
          </p>
          <SearchBar />
        </div>
      </div>
    </div>
  );
}