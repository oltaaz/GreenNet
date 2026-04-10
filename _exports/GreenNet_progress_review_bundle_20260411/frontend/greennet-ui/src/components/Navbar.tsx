import { Link, NavLink } from "react-router-dom";
import { useBackendStatus } from "../hooks/useBackendStatus";

const navItems = [
  { to: "/", label: "Home" },
  { to: "/dashboard", label: "Dashboard" },
  { to: "/results", label: "Results" },
  { to: "/simulator", label: "Simulator" },
  { to: "/about", label: "About" },
];

export default function Navbar() {
  const { status } = useBackendStatus();
  const indicatorLabel =
    status === "online" ? "Backend connected" : status === "checking" ? "Checking backend" : "Backend offline";

  return (
    <header className="top-nav glass-card">
      <div className="top-nav-inner">
        <Link to="/" className="brand" aria-label="Go to GreenNet home">
          <span className="brand-mark" aria-hidden>
            <svg viewBox="0 0 24 24">
              <path d="M12 1.75c5.1 2.1 8.5 7 8.5 12.2A8.26 8.26 0 0 1 12.25 22 8.26 8.26 0 0 1 4 13.95c0-5.2 3.2-10.1 8-12.2Z" />
            </svg>
          </span>
          <span>
            <strong>GreenNet</strong>
            <small>AI-Driven Sustainable Networking</small>
          </span>
        </Link>

        <nav className="top-links" aria-label="Primary">
          <span className={`backend-indicator ${status}`.trim()}>
            <span className="backend-indicator-dot" aria-hidden />
            {indicatorLabel}
          </span>
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => (isActive ? "nav-link active" : "nav-link")}
              end={item.to === "/"}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}
