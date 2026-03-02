import { Link } from "react-router-dom";

export default function HomePage() {
  return (
    <div className="page home-page">
      <section className="hero-block glass-card home-hero">
        <p className="page-eyebrow">GreenNet</p>
        <h1>GreenNet</h1>
        <h2>AI-Driven Sustainable Networking</h2>
        <p>Adaptive policies cut network energy while keeping service quality stable.</p>
        <div className="button-row">
          <Link className="btn-primary" to="/dashboard">
            Open Dashboard
          </Link>
          <Link className="btn-muted" to="/simulator">
            Open Simulator
          </Link>
        </div>
      </section>

      <section className="info-grid">
        <article className="glass-card info-card">
          <h3>Problem</h3>
          <p>
            Always-on links waste power at low traffic. Static rules miss changing network conditions.
          </p>
        </article>

        <article className="glass-card info-card">
          <h3>Our Solution</h3>
          <p>
            GreenNet learns when links should turn ON or OFF per step using QoS-aware rewards.
          </p>
        </article>

        <article className="glass-card info-card">
          <h3>Architecture</h3>
          <p>
            Python APIs provide run and step metrics. React + Vite presents dashboards, charts, and simulation playback.
          </p>
        </article>

        <article className="glass-card info-card">
          <h3>Results Highlights</h3>
          <p>
            Track energy, carbon, delay, drops, and active ratio across policies in one workflow.
          </p>
        </article>
      </section>
    </div>
  );
}
