export default function AboutPage() {
  return (
    <div className="page static-page">
      <section className="page-title-row">
        <div>
          <p className="page-eyebrow">About</p>
          <h1>GreenNet: AI-Driven Sustainable Networking</h1>
        </div>
      </section>

      <section className="glass-card prose-card">
        <p>
          GreenNet is a capstone project focused on reducing network energy consumption through adaptive control
          policies. Instead of keeping every link active, GreenNet learns when links can be turned OFF while preserving
          quality-of-service constraints.
        </p>
        <p>
          This production-oriented frontend is designed for thesis demos and evaluation workflows. It combines
          interactive KPI dashboards, per-policy comparison overlays, and a real-time simulator canvas that visualizes
          packet movement under dynamic link states.
        </p>
        <p>
          The app is API-first: all pages consume backend run data using a single typed client layer (`src/lib/api.ts`)
          and gracefully fallback when optional endpoints (topology link-state packet-events) are not yet available.
        </p>
      </section>
    </div>
  );
}
