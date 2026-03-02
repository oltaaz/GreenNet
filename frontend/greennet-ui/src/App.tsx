import { Navigate, Route, Routes } from "react-router-dom";
import Navbar from "./components/Navbar";
import AboutPage from "./pages/AboutPage";
import ComparePage from "./pages/ComparePage";
import DashboardPage from "./pages/DashboardPage";
import HomePage from "./pages/HomePage";
import ResearchPage from "./pages/ResearchPage";
import SimulatorPage from "./pages/SimulatorPage";
import "./styles/app.css";

export default function App() {
  return (
    <div className="app-shell">
      <div className="bg-layer" aria-hidden />
      <div className="bg-motion-lines" aria-hidden />
      <div className="bg-motion-lines-fade" aria-hidden />
      <div className="side-circuit side-circuit-left" aria-hidden />
      <div className="side-circuit side-circuit-right" aria-hidden />
      <Navbar />
      <main className="page-shell">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/simulator" element={<SimulatorPage />} />
          <Route path="/research" element={<ResearchPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
