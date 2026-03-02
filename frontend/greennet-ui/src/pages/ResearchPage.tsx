export default function ResearchPage() {
  return (
    <div className="page static-page">
      <section className="page-title-row">
        <div>
          <p className="page-eyebrow">Research</p>
          <h1>Methodology and Evaluation Design</h1>
        </div>
      </section>

      <section className="info-grid">
        <article className="glass-card info-card">
          <h3>Objective Function</h3>
          <p>
            The policy optimizes a reward balancing energy reduction, packet loss penalties, and QoS constraints.
            Per-step metrics are accumulated to evaluate episode-level performance.
          </p>
        </article>

        <article className="glass-card info-card">
          <h3>Simulation Inputs</h3>
          <p>
            Experiments vary policy type, scenario profile, random seed, and episode steps. This frontend keeps these
            controls synchronized across dashboard, compare, and simulator views.
          </p>
        </article>

        <article className="glass-card info-card">
          <h3>Core Metrics</h3>
          <p>
            Energy (kWh), carbon (g CO2), delay (ms), delivered, dropped, active ratio, and reward are reported at each
            step and summarized at run-level in comparison tables.
          </p>
        </article>

        <article className="glass-card info-card">
          <h3>Interpretation</h3>
          <p>
            Strong policies reduce energy and carbon without severe rise in dropped traffic or delay. Topology playback
            validates whether aggressive link shutdown decisions align with delivery outcomes.
          </p>
        </article>
      </section>
    </div>
  );
}
