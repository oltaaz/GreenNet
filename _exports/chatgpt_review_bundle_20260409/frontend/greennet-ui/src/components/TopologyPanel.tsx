import { useEffect, useMemo, useState } from "react";
import { edgeId, withLayout } from "../lib/data";
import type { LinkStateMap, StepMetrics, TopologyData } from "../lib/types";

type TopologyPanelProps = {
  topology: TopologyData;
  linkState?: LinkStateMap;
  previousLinkState?: LinkStateMap;
  stepIndex?: number;
  metrics?: StepMetrics;
  previousMetrics?: StepMetrics;
  title?: string;
  subtitle?: string;
};

type EdgeDraw = {
  id: string;
  sourceId: string;
  targetId: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  on: boolean;
  switched: boolean;
  utilization: number;
  powerW: number;
};

type PacketDot = {
  id: string;
  x: number;
  y: number;
  kind: "delivered" | "rerouted" | "dropped";
  size: number;
};

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function clampInt(value: number, min: number, max: number): number {
  return Math.round(clamp(value, min, max));
}

function noiseFrom(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0) / 4294967295;
}

function pickEdge(edges: EdgeDraw[], seed: string, index: number): EdgeDraw {
  const n = noiseFrom(`${seed}-${index}`);
  return edges[Math.floor(n * edges.length) % edges.length];
}

function isLinkOn(linkState: LinkStateMap | undefined, source: string, target: string): boolean {
  if (!linkState) {
    return true;
  }

  const id = edgeId(source, target);
  if (id in linkState) {
    return Boolean(linkState[id]);
  }
  return true;
}

export default function TopologyPanel({
  topology,
  linkState,
  previousLinkState,
  stepIndex = 0,
  metrics,
  previousMetrics,
  title = "Network Topology",
  subtitle = "Simulated packet flow and real link-state playback. Switching links can trigger reroutes or drops.",
}: TopologyPanelProps) {
  const [phase, setPhase] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setPhase((prev) => (prev + 0.018) % 1);
    }, 33);

    return () => clearInterval(timer);
  }, []);

  const graph = useMemo(() => withLayout(topology), [topology]);
  const nodeMap = useMemo(() => new Map(graph.nodes.map((node) => [node.id, node])), [graph.nodes]);

  const edges = useMemo(() => {
    const flowPressure = clamp(
      ((metrics?.delivered ?? 0) + (metrics?.dropped ?? 0)) / Math.max(8, graph.edges.length * 14),
      0,
      1,
    );

    return graph.edges
      .map((edge) => {
        const source = nodeMap.get(edge.source);
        const target = nodeMap.get(edge.target);

        if (!source || !target || source.x == null || source.y == null || target.x == null || target.y == null) {
          return null;
        }

        const on = isLinkOn(linkState, edge.source, edge.target);
        const previousOn = isLinkOn(previousLinkState, edge.source, edge.target);

        const n = noiseFrom(`${edge.id}-${stepIndex}`);
        const utilization = on
          ? clamp(0.2 + n * 0.55 + flowPressure * 0.45, 0.15, 1)
          : clamp(0.03 + n * 0.14, 0.02, 0.2);

        return {
          id: edge.id,
          sourceId: edge.source,
          targetId: edge.target,
          x1: source.x * 100,
          y1: source.y * 70,
          x2: target.x * 100,
          y2: target.y * 70,
          on,
          switched: on !== previousOn,
          utilization,
          powerW: on ? 8 + utilization * 44 : 0.8,
        } satisfies EdgeDraw;
      })
      .filter((edge): edge is EdgeDraw => edge !== null);
  }, [graph.edges, linkState, metrics?.delivered, metrics?.dropped, nodeMap, previousLinkState, stepIndex]);

  const activeEdges = useMemo(() => edges.filter((edge) => edge.on), [edges]);
  const inactiveEdges = useMemo(() => edges.filter((edge) => !edge.on), [edges]);
  const switchedCount = useMemo(() => edges.filter((edge) => edge.switched).length, [edges]);

  const packets = useMemo(() => {
    const output: PacketDot[] = [];
    if (edges.length === 0) {
      return output;
    }

    const deliveredStep = Math.max(0, metrics?.delivered ?? 0);
    const droppedStep = Math.max(0, metrics?.dropped ?? 0);

    const deliveredCount = clampInt(
      deliveredStep > 0 ? deliveredStep / 3 : Math.max(2, activeEdges.length * 0.6),
      2,
      20,
    );
    const droppedCount = clampInt(droppedStep / 2, 0, 12);

    const reroutedCount =
      droppedCount > 0 && inactiveEdges.length > 0 && activeEdges.length > 0
        ? Math.min(droppedCount, Math.ceil(droppedCount * 0.55))
        : 0;
    const hardDropCount = droppedCount - reroutedCount;

    for (let index = 0; index < deliveredCount; index += 1) {
      const edge = pickEdge(activeEdges.length > 0 ? activeEdges : edges, `ok-${stepIndex}`, index);
      const speedFactor = 0.65 + noiseFrom(`${edge.id}-spd-${index}`) * 0.95;
      const offset = noiseFrom(`${edge.id}-off-${index}`);
      const progress = (phase * speedFactor + offset) % 1;

      output.push({
        id: `ok-${edge.id}-${index}`,
        x: edge.x1 + (edge.x2 - edge.x1) * progress,
        y: edge.y1 + (edge.y2 - edge.y1) * progress,
        kind: "delivered",
        size: 0.65 + edge.utilization * 0.9,
      });
    }

    for (let index = 0; index < reroutedCount; index += 1) {
      const edge = pickEdge(activeEdges.length > 0 ? activeEdges : edges, `reroute-${stepIndex}`, index);
      const progress = (phase * 0.9 + noiseFrom(`${edge.id}-rr-${index}`) + index * 0.09) % 1;

      output.push({
        id: `reroute-${edge.id}-${index}`,
        x: edge.x1 + (edge.x2 - edge.x1) * progress,
        y: edge.y1 + (edge.y2 - edge.y1) * progress,
        kind: "rerouted",
        size: 0.74,
      });
    }

    for (let index = 0; index < hardDropCount; index += 1) {
      const edge = pickEdge(inactiveEdges.length > 0 ? inactiveEdges : edges, `drop-${stepIndex}`, index);
      const progress = (phase * 0.72 + noiseFrom(`${edge.id}-dr-${index}`) + index * 0.11) % 1;

      output.push({
        id: `drop-${edge.id}-${index}`,
        x: edge.x1 + (edge.x2 - edge.x1) * progress,
        y: edge.y1 + (edge.y2 - edge.y1) * progress,
        kind: "dropped",
        size: 0.8,
      });
    }

    return output;
  }, [activeEdges, edges, inactiveEdges, metrics?.delivered, metrics?.dropped, phase, previousMetrics?.delivered, previousMetrics?.dropped, stepIndex]);

  const nodeLoad = useMemo(() => {
    const loadMap = new Map<string, number>();
    for (const node of graph.nodes) {
      loadMap.set(node.id, 0);
    }

    for (const edge of edges) {
      if (!edge.on) {
        continue;
      }
      loadMap.set(edge.sourceId, (loadMap.get(edge.sourceId) ?? 0) + edge.utilization * 0.5);
      loadMap.set(edge.targetId, (loadMap.get(edge.targetId) ?? 0) + edge.utilization * 0.5);
    }

    const peak = Math.max(0.01, ...loadMap.values());
    return Object.fromEntries([...loadMap.entries()].map(([nodeId, load]) => [nodeId, clamp(load / peak, 0, 1)]));
  }, [edges, graph.nodes]);

  const estimatedPower = useMemo(() => edges.reduce((sum, edge) => sum + edge.powerW, 0), [edges]);
  const energyDelta = (metrics?.energy_kwh ?? 0) - (previousMetrics?.energy_kwh ?? 0);

  return (
    <section className="glass-card topology-card">
      <div className="card-heading">
        <p>Topology</p>
        <h3>{title}</h3>
      </div>
      <p className="card-caption">{subtitle}</p>

      <div className="topology-stats">
        <article>
          <span>ON Links</span>
          <strong>{activeEdges.length}</strong>
        </article>
        <article>
          <span>OFF Links</span>
          <strong>{inactiveEdges.length}</strong>
        </article>
        <article>
          <span>Switches</span>
          <strong>{switchedCount}</strong>
        </article>
        <article>
          <span>Step Energy</span>
          <strong>{(metrics?.energy_kwh ?? 0).toFixed(3)} kWh</strong>
          <small className={energyDelta >= 0 ? "hot" : "cool"}>
            {energyDelta >= 0 ? "+" : ""}
            {energyDelta.toFixed(3)}
          </small>
        </article>
        <article>
          <span>Link Power (visual est.)</span>
          <strong>{estimatedPower.toFixed(1)} W</strong>
        </article>
      </div>

      <svg viewBox="0 0 100 70" className="topology-svg" role="img" aria-label="Network topology graph">
        <defs>
          <filter id="edgeGlowStrong">
            <feGaussianBlur stdDeviation="1.2" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <radialGradient id="nodeHalo" r="65%">
            <stop offset="0%" stopColor="rgba(85,255,170,0.7)" />
            <stop offset="100%" stopColor="rgba(85,255,170,0)" />
          </radialGradient>
        </defs>

        {edges.map((edge) => (
          <g key={edge.id}>
            <line
              x1={edge.x1}
              y1={edge.y1}
              x2={edge.x2}
              y2={edge.y2}
              stroke={edge.on ? `rgba(85,255,170,${0.42 + edge.utilization * 0.58})` : "rgba(108,126,149,0.36)"}
              strokeWidth={edge.on ? 0.7 + edge.utilization * 1.45 : 1.1}
              strokeDasharray={edge.on ? undefined : "2.4 2.2"}
              filter={edge.on ? "url(#edgeGlowStrong)" : undefined}
            />
            {edge.switched ? (
              <line
                x1={edge.x1}
                y1={edge.y1}
                x2={edge.x2}
                y2={edge.y2}
                stroke="rgba(255,190,85,0.9)"
                strokeWidth={0.9}
                strokeDasharray="1.1 1.4"
              />
            ) : null}
          </g>
        ))}

        {packets.map((packet) => {
          const fill =
            packet.kind === "delivered" ? "#6bffbe" : packet.kind === "rerouted" ? "#65dcff" : "#ff637a";

          return (
            <circle
              key={packet.id}
              cx={packet.x}
              cy={packet.y}
              r={packet.size}
              fill={fill}
              opacity={0.94}
              filter="url(#edgeGlowStrong)"
            />
          );
        })}

        {graph.nodes.map((node) => {
          const x = (node.x ?? 0.5) * 100;
          const y = (node.y ?? 0.5) * 70;
          const load = nodeLoad[node.id] ?? 0;
          const active = load > 0.02;

          return (
            <g key={node.id}>
              {active ? <circle cx={x} cy={y} r={3.2 + load * 1.9} fill="url(#nodeHalo)" /> : null}
              <circle
                cx={x}
                cy={y}
                r={1.2 + load * 1.1}
                fill={active ? "#58ffaf" : "#607590"}
                stroke={active ? "rgba(160,255,213,0.66)" : "rgba(137,154,178,0.4)"}
                strokeWidth={0.32}
              />
              <text x={x + 1.1} y={y - 1.15} className="topology-node-label">
                {node.label ?? node.id}
              </text>
            </g>
          );
        })}
      </svg>

      <div className="topology-legend">
        <span>
          <i className="dot on" /> ON link
        </span>
        <span>
          <i className="dot off" /> OFF link
        </span>
        <span>
          <i className="dot rerouted" /> Rerouted packet
        </span>
        <span>
          <i className="dot dropped" /> Dropped packet
        </span>
      </div>
    </section>
  );
}
