import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { LandingPage } from './components/LandingPage';
import { MediumSummaryPage } from './components/MediumSummaryPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<LandingPage />} />
          <Route path="medium/:id" element={<MediumSummaryPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}