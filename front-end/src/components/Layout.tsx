import React from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { SearchBar } from './SearchBar';

export function Layout() {
  const location = useLocation();
  const isLandingPage = location.pathname === '/';

  return (
    <div className="min-h-screen">
      {!isLandingPage && (
        <div className="navbar bg-base-100 shadow-md sticky top-0 z-40">
          <div className="flex-1 justify-center max-w-4xl mx-auto w-full px-4">
            <SearchBar />
          </div>
        </div>
      )}
      <Outlet />
    </div>
  );
}