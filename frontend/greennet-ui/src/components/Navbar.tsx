import { Link, NavLink } from "react-router-dom";

const navItems = [
  { to: "/", label: "Home" },
  { to: "/dashboard", label: "Dashboard" },
  { to: "/compare", label: "Compare" },
  { to: "/simulator", label: "Simulator" },
  { to: "/research", label: "Research" },
  { to: "/about", label: "About" },
];

export default function Navbar() {
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
