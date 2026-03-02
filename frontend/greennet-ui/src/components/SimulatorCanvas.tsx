import { useEffect, useMemo, useRef } from "react";
import { edgeId, withLayout } from "../lib/data";
import type { StepState, TopologyData } from "../lib/types";

type SimulatorCanvasProps = {
  topology: TopologyData;
  steps: StepState[];
  currentStep: number;
  playing: boolean;
  speed: number;
  showDropped: boolean;
  onStepChange: (step: number) => void;
};

type RenderEdge = {
  id: string;
  sourceId: string;
  targetId: string;
  sourceX: number;
  sourceY: number;
  targetX: number;
  targetY: number;
  on: boolean;
};

type PacketKind = "delivered" | "rerouted" | "dropped";

type RenderPacket = {
  edge: RenderEdge;
  progress: number;
  size: number;
  kind: PacketKind;
  reverse: boolean;
};

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function seeded(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function hash(value: string): number {
  let h = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    h ^= value.charCodeAt(index);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function noiseFrom(value: string): number {
  return hash(value) / 4294967295;
}

function pickEdges(edges: RenderEdge[], count: number, seedKey: string): RenderEdge[] {
  if (!edges.length || count <= 0) {
    return [];
  }

  const rng = seeded(hash(seedKey));
  const results: RenderEdge[] = [];
  for (let idx = 0; idx < count; idx += 1) {
    const edge = edges[Math.floor(rng() * edges.length) % edges.length];
    results.push(edge);
  }
  return results;
}

function packetCountFromDelta(delta: number, fallback: number, minCount: number, maxCount: number): number {
  if (!Number.isFinite(delta)) {
    return fallback;
  }
  if (delta <= 0) {
    return fallback;
  }
  return Math.max(minCount, Math.min(maxCount, Math.round(delta / 3)));
}

export default function SimulatorCanvas({
  topology,
  steps,
  currentStep,
  playing,
  speed,
  showDropped,
  onStepChange,
}: SimulatorCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const onStepChangeRef = useRef(onStepChange);
  const stepRef = useRef(currentStep);
  const phaseRef = useRef(0);
  const sizeRef = useRef({ width: 1000, height: 580 });

  const graph = useMemo(() => withLayout(topology), [topology]);

  const renderEdges = useMemo(() => {
    const nodeMap = new Map(graph.nodes.map((node) => [node.id, node]));
    const current = steps[currentStep] ?? steps[0];

    return graph.edges
      .map((edge) => {
        const source = nodeMap.get(edge.source);
        const target = nodeMap.get(edge.target);
        if (!source || !target || source.x == null || source.y == null || target.x == null || target.y == null) {
          return null;
        }

        const linkKey = edge.id || edgeId(edge.source, edge.target);
        const on = current?.links_on ? Boolean(current.links_on[linkKey] ?? current.links_on[edgeId(edge.source, edge.target)]) : true;

        return {
          id: linkKey,
          sourceId: edge.source,
          targetId: edge.target,
          sourceX: source.x,
          sourceY: source.y,
          targetX: target.x,
          targetY: target.y,
          on,
        };
      })
      .filter((edge): edge is RenderEdge => edge !== null);
  }, [currentStep, graph.edges, graph.nodes, steps]);

  useEffect(() => {
    onStepChangeRef.current = onStepChange;
  }, [onStepChange]);

  useEffect(() => {
    stepRef.current = currentStep;
    phaseRef.current = 0;
  }, [currentStep]);

  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) {
      return;
    }

    const observer = new ResizeObserver(() => {
      const bounds = container.getBoundingClientRect();
      sizeRef.current = {
        width: Math.max(320, Math.floor(bounds.width)),
        height: Math.max(280, Math.floor(bounds.height)),
      };
    });

    observer.observe(container);

    const bounds = container.getBoundingClientRect();
    sizeRef.current = {
      width: Math.max(320, Math.floor(bounds.width)),
      height: Math.max(280, Math.floor(bounds.height)),
    };

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || steps.length === 0) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    let frame = 0;
    let lastTimestamp = 0;

    const draw = (stepIndex: number, phase: number): void => {
      const { width, height } = sizeRef.current;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(dpr, 0, 0, dpr, 0, 0);

      const step = steps[stepIndex];
      const previous = steps[(stepIndex - 1 + steps.length) % steps.length];

      context.clearRect(0, 0, width, height);

      const background = context.createLinearGradient(0, 0, width, height);
      background.addColorStop(0, "#041024");
      background.addColorStop(1, "#081a2f");
      context.fillStyle = background;
      context.fillRect(0, 0, width, height);

      const haze = context.createRadialGradient(width * 0.18, height * 0.12, 20, width * 0.18, height * 0.12, width * 0.8);
      haze.addColorStop(0, "rgba(85,255,170,0.1)");
      haze.addColorStop(1, "rgba(85,255,170,0)");
      context.fillStyle = haze;
      context.fillRect(0, 0, width, height);

      context.strokeStyle = "rgba(73, 104, 140, 0.16)";
      context.lineWidth = 1;
      for (let x = 0; x < width; x += 36) {
        context.beginPath();
        context.moveTo(x, 0);
        context.lineTo(x, height);
        context.stroke();
      }
      for (let y = 0; y < height; y += 36) {
        context.beginPath();
        context.moveTo(0, y);
        context.lineTo(width, y);
        context.stroke();
      }

      const trafficPressure = clamp(
        (step.metrics.delivered + step.metrics.dropped) / Math.max(8, renderEdges.length * 14),
        0,
        1,
      );

      const edges = renderEdges.map((edge) => {
        const linkKey = edge.id;
        const on = step.links_on ? Boolean(step.links_on[linkKey] ?? step.links_on[edge.id]) : edge.on;
        const prevOn = previous.links_on
          ? Boolean(previous.links_on[linkKey] ?? previous.links_on[edge.id])
          : edge.on;

        const noise = noiseFrom(`${edge.id}-${step.t}`);
        const utilization = on
          ? clamp(0.2 + noise * 0.55 + trafficPressure * 0.45, 0.15, 1)
          : clamp(0.03 + noise * 0.14, 0.02, 0.22);

        return {
          ...edge,
          on,
          switched: on !== prevOn,
          utilization,
        };
      });

      for (const edge of edges) {
        const x1 = edge.sourceX * width;
        const y1 = edge.sourceY * height;
        const x2 = edge.targetX * width;
        const y2 = edge.targetY * height;

        context.setLineDash(edge.on ? [] : [4, 4]);
        context.strokeStyle = edge.on ? `rgba(85,255,170,${0.36 + edge.utilization * 0.58})` : "rgba(113,131,153,0.38)";
        context.lineWidth = edge.on ? 1.4 + edge.utilization * 2.6 : 1.4;

        context.beginPath();
        context.moveTo(x1, y1);
        context.lineTo(x2, y2);
        context.stroke();

        if (edge.switched) {
          context.setLineDash([2, 3]);
          context.strokeStyle = "rgba(255,198,96,0.95)";
          context.lineWidth = 1.4;
          context.beginPath();
          context.moveTo(x1, y1);
          context.lineTo(x2, y2);
          context.stroke();
        }
      }
      context.setLineDash([]);

      const activeEdges = edges.filter((edge) => edge.on);
      const inactiveEdges = edges.filter((edge) => !edge.on);

      const deliveredDelta = Math.max(0, step.metrics.delivered - previous.metrics.delivered);
      const droppedDelta = Math.max(0, step.metrics.dropped - previous.metrics.dropped);

      const packets: RenderPacket[] = [];

      if (Array.isArray(step.packet_events) && step.packet_events.length > 0) {
        for (const [index, packet] of step.packet_events.slice(0, 50).entries()) {
          const packetEdgeId = packet.edge_id ?? (packet.source && packet.target ? edgeId(packet.source, packet.target) : undefined);
          const edge = edges.find((candidate) => candidate.id === packetEdgeId) ?? edges[index % Math.max(edges.length, 1)];
          if (!edge) {
            continue;
          }

          const status = String(packet.status ?? "in_transit").toLowerCase();
          const kind: PacketKind = status === "dropped" ? "dropped" : status === "rerouted" ? "rerouted" : "delivered";
          if (!showDropped && kind === "dropped") {
            continue;
          }

          packets.push({
            edge,
            progress: typeof packet.progress === "number" ? packet.progress : (phase + index * 0.03) % 1,
            kind,
            reverse: index % 2 === 0,
            size: kind === "delivered" ? 2.4 : 2.1,
          });
        }
      } else {
        const deliveredEdges = pickEdges(
          activeEdges.length > 0 ? activeEdges : edges,
          packetCountFromDelta(deliveredDelta, Math.max(3, Math.round(activeEdges.length * 0.45)), 3, 28),
          `${step.t}-ok`,
        );

        const droppedCount = packetCountFromDelta(droppedDelta, 0, 0, 14);
        const reroutedCount =
          inactiveEdges.length > 0 && activeEdges.length > 0 ? Math.min(droppedCount, Math.ceil(droppedCount * 0.55)) : 0;
        const hardDropCount = droppedCount - reroutedCount;

        const rerouteEdges = pickEdges(activeEdges.length > 0 ? activeEdges : edges, reroutedCount, `${step.t}-reroute`);
        const droppedEdges = pickEdges(
          inactiveEdges.length > 0 ? inactiveEdges : edges,
          hardDropCount,
          `${step.t}-drop`,
        );

        for (const [index, edge] of deliveredEdges.entries()) {
          packets.push({
            edge,
            progress: (phase * (0.76 + noiseFrom(`${edge.id}-spd-${index}`)) + (index * 0.15) % 1) % 1,
            kind: "delivered",
            reverse: index % 2 === 1,
            size: 2.45,
          });
        }

        for (const [index, edge] of rerouteEdges.entries()) {
          packets.push({
            edge,
            progress: (phase * 0.95 + noiseFrom(`${edge.id}-r-${index}`) + index * 0.09) % 1,
            kind: "rerouted",
            reverse: index % 2 === 0,
            size: 2.2,
          });
        }

        if (showDropped) {
          for (const [index, edge] of droppedEdges.entries()) {
            packets.push({
              edge,
              progress: (phase * 0.67 + noiseFrom(`${edge.id}-d-${index}`) + index * 0.12) % 1,
              kind: "dropped",
              reverse: index % 2 === 0,
              size: 2.3,
            });
          }
        }
      }

      for (const packet of packets) {
        const progress = packet.reverse ? 1 - packet.progress : packet.progress;
        const prevProgress = clamp(progress - 0.055, 0, 1);

        const x = (packet.edge.sourceX + (packet.edge.targetX - packet.edge.sourceX) * progress) * width;
        const y = (packet.edge.sourceY + (packet.edge.targetY - packet.edge.sourceY) * progress) * height;

        const tx = (packet.edge.sourceX + (packet.edge.targetX - packet.edge.sourceX) * prevProgress) * width;
        const ty = (packet.edge.sourceY + (packet.edge.targetY - packet.edge.sourceY) * prevProgress) * height;

        const color = packet.kind === "delivered" ? "#65ffbf" : packet.kind === "rerouted" ? "#66dcff" : "#ff6075";

        context.strokeStyle = packet.kind === "dropped" ? "rgba(255,96,117,0.45)" : "rgba(102,255,202,0.3)";
        context.lineWidth = 1.3;
        context.beginPath();
        context.moveTo(tx, ty);
        context.lineTo(x, y);
        context.stroke();

        context.fillStyle = color;
        context.shadowBlur = 12;
        context.shadowColor =
          packet.kind === "dropped" ? "rgba(255,96,117,0.66)" : packet.kind === "rerouted" ? "rgba(102,220,255,0.66)" : "rgba(85,255,170,0.8)";
        context.beginPath();
        context.arc(x, y, packet.size, 0, Math.PI * 2);
        context.fill();
      }
      context.shadowBlur = 0;

      const nodeLoad = new Map<string, number>();
      for (const node of graph.nodes) {
        nodeLoad.set(node.id, 0);
      }
      for (const edge of edges) {
        if (!edge.on) {
          continue;
        }
        nodeLoad.set(edge.sourceId, (nodeLoad.get(edge.sourceId) ?? 0) + edge.utilization * 0.5);
        nodeLoad.set(edge.targetId, (nodeLoad.get(edge.targetId) ?? 0) + edge.utilization * 0.5);
      }
      const peakLoad = Math.max(0.1, ...nodeLoad.values());

      for (const node of graph.nodes) {
        const x = (node.x ?? 0.5) * width;
        const y = (node.y ?? 0.5) * height;

        const normalizedLoad = clamp((nodeLoad.get(node.id) ?? 0) / peakLoad, 0, 1);
        const active = normalizedLoad > 0.02;

        context.fillStyle = active ? "rgba(85,255,170,0.2)" : "rgba(109,129,154,0.18)";
        context.beginPath();
        context.arc(x, y, 7 + normalizedLoad * 4, 0, Math.PI * 2);
        context.fill();

        context.fillStyle = active ? "rgba(85,255,170,0.98)" : "rgba(131,151,175,0.9)";
        context.beginPath();
        context.arc(x, y, 3.2 + normalizedLoad * 2.2, 0, Math.PI * 2);
        context.fill();

        context.fillStyle = "rgba(207,226,247,0.95)";
        context.font = "11px Inter, system-ui";
        context.fillText(node.label ?? node.id, x + 7, y - 7);
      }

      const onCount = activeEdges.length;
      const offCount = inactiveEdges.length;
      const switchedCount = edges.filter((edge) => edge.switched).length;
      const energyDelta = step.metrics.energy_kwh - previous.metrics.energy_kwh;

      const hudX = 14;
      const hudY = 14;
      const hudW = 294;
      const hudH = 102;

      context.fillStyle = "rgba(4,16,33,0.75)";
      context.strokeStyle = "rgba(120,154,190,0.46)";
      context.lineWidth = 1;
      context.beginPath();
      context.roundRect(hudX, hudY, hudW, hudH, 10);
      context.fill();
      context.stroke();

      context.fillStyle = "rgba(218,236,255,0.98)";
      context.font = "600 13px Inter, system-ui";
      context.fillText(`Step ${step.t}  |  Active ${Math.round(step.metrics.active_ratio * 100)}%`, hudX + 10, hudY + 22);

      context.font = "12px Inter, system-ui";
      context.fillStyle = "rgba(184,206,232,0.96)";
      context.fillText(`Links ON ${onCount} / OFF ${offCount} / switched ${switchedCount}`, hudX + 10, hudY + 42);
      context.fillText(
        `Energy ${step.metrics.energy_kwh.toFixed(3)} kWh (${energyDelta >= 0 ? "+" : ""}${energyDelta.toFixed(3)})`,
        hudX + 10,
        hudY + 62,
      );
      context.fillText(
        `Delivered +${deliveredDelta.toFixed(0)} | Dropped +${droppedDelta.toFixed(0)} | Carbon ${step.metrics.carbon_g.toFixed(2)} g`,
        hudX + 10,
        hudY + 82,
      );

      const meterX = width - 28;
      const meterY = 18;
      const meterH = 112;
      const meterW = 10;

      context.fillStyle = "rgba(113,131,154,0.34)";
      context.fillRect(meterX, meterY, meterW, meterH);

      const meterFill = clamp(step.metrics.active_ratio, 0, 1);
      context.fillStyle = "rgba(85,255,170,0.9)";
      context.fillRect(meterX, meterY + meterH * (1 - meterFill), meterW, meterH * meterFill);

      context.fillStyle = "rgba(179,205,235,0.95)";
      context.font = "11px Inter, system-ui";
      context.fillText("Link Power", width - 72, meterY + meterH + 16);

      context.fillStyle = "rgba(4,16,33,0.68)";
      context.beginPath();
      context.roundRect(14, height - 34, 264, 20, 8);
      context.fill();

      context.font = "11px Inter, system-ui";
      context.fillStyle = "#65ffbf";
      context.fillText("Delivered", 24, height - 20);
      context.fillStyle = "#66dcff";
      context.fillText("Rerouted", 92, height - 20);
      context.fillStyle = "#ff6075";
      context.fillText("Dropped", 160, height - 20);
      context.fillStyle = "rgba(255,198,96,0.95)";
      context.fillText("Switch event", 214, height - 20);
    };

    const loop = (timestamp: number): void => {
      if (!lastTimestamp) {
        lastTimestamp = timestamp;
      }
      const deltaSeconds = Math.max(0, (timestamp - lastTimestamp) / 1000);
      lastTimestamp = timestamp;

      if (playing && steps.length > 1) {
        phaseRef.current += deltaSeconds * speed * 0.85;
        if (phaseRef.current >= 1) {
          const jump = Math.floor(phaseRef.current);
          phaseRef.current -= jump;
          const next = (stepRef.current + jump) % steps.length;
          stepRef.current = next;
          onStepChangeRef.current(next);
        }
      }

      draw(stepRef.current, phaseRef.current);
      frame = requestAnimationFrame(loop);
    };

    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [graph.nodes, playing, renderEdges, showDropped, speed, steps]);

  return (
    <div className="sim-canvas-wrap" ref={containerRef}>
      <canvas ref={canvasRef} className="sim-canvas" />
    </div>
  );
}
