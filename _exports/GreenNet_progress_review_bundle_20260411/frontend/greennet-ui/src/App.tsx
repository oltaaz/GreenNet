import { Navigate, Route, Routes } from "react-router-dom";
import Navbar from "./components/Navbar";
import { ErrorNotice, InfoNotice } from "./components/StatusState";
import { useBackendStatus } from "./hooks/useBackendStatus";
import AboutPage from "./pages/AboutPage";
import ComparePage from "./pages/ComparePage";
import DashboardPage from "./pages/DashboardPage";
import HomePage from "./pages/HomePage";
import SimulatorPage from "./pages/SimulatorPage";
import "./styles/app.css";

export default function App() {
  const { status, expectedBackendUrl, refresh } = useBackendStatus();

  return (
    <div className="app-shell">
      <div className="bg-layer" aria-hidden />
      <div className="bg-motion-lines" aria-hidden />
      <div className="bg-motion-lines-fade" aria-hidden />
      <div className="side-circuit side-circuit-left" aria-hidden />
      <div className="side-circuit side-circuit-right" aria-hidden />
      <Navbar />
      <main className="page-shell">
        {status === "offline" ? (
          <>
            <ErrorNotice
              title="Backend API unavailable"
              description={`Expected GreenNet FastAPI at ${expectedBackendUrl}. Start the backend first, then refresh this page.`}
            />
            <div className="backend-banner-actions">
              <button className="btn-muted" onClick={() => void refresh()}>
                Retry backend check
              </button>
            </div>
          </>
        ) : null}
        {status === "checking" ? (
          <InfoNotice
            title="Checking backend connectivity"
            description="Verifying that the local GreenNet FastAPI backend is available before enabling live run actions."
          />
        ) : null}
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/results" element={<ComparePage />} />
          <Route path="/compare" element={<Navigate to="/results" replace />} />
          <Route path="/simulator" element={<SimulatorPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
