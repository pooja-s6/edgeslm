import { useEffect, useState } from "react";
import { api, type HardwareSnapshot } from "../api/client";

interface Props {
  pollMs?: number;
}

function Meter({
  label,
  value,
  max,
  unit,
  tone = "teal",
}: {
  label: string;
  value: number;
  max: number;
  unit: string;
  tone?: "teal" | "amber" | "slate";
}) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className="meter">
      <div className="meter-head">
        <span>{label}</span>
        <span className="mono">
          {value.toFixed(1)}
          {unit}
        </span>
      </div>
      <div className="meter-track">
        <div
          className={`meter-fill meter-fill--${tone}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export default function HardwareMonitor({ pollMs = 2000 }: Props) {
  const [snap, setSnap] = useState<HardwareSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const tick = async () => {
      try {
        const data = await api.hardware();
        if (!cancelled) {
          setSnap(data);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Hardware poll failed");
        }
      }
    };

    void tick();
    const id = window.setInterval(() => void tick(), pollMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [pollMs]);

  return (
    <section className="panel panel--live">
      <header className="panel-header">
        <div className="panel-kicker">Telemetry</div>
        <h2>Hardware Monitor</h2>
        <p>Live CPU utilization, system memory, and Ollama runner RSS</p>
      </header>

      {error && <p className="status status--warn">{error}</p>}

      {!snap && !error && <p className="status">Polling telemetry…</p>}

      {snap && (
        <div className="meter-stack">
          <Meter
            label="CPU"
            value={snap.cpu_percent}
            max={100}
            unit="%"
            tone="amber"
          />
          <Meter
            label="System RAM used"
            value={snap.used_ram_mb}
            max={snap.total_ram_mb}
            unit=" MB"
            tone="teal"
          />
          <Meter
            label="Available RAM"
            value={snap.available_ram_mb}
            max={snap.total_ram_mb}
            unit=" MB"
            tone="slate"
          />
          <Meter
            label="Ollama runner RSS"
            value={snap.runner_rss_mb}
            max={Math.max(snap.total_ram_mb * 0.35, snap.runner_rss_mb || 1)}
            unit=" MB"
            tone="amber"
          />
          <div className="hw-meta mono">
            <span>{snap.total_ram_mb.toFixed(0)} MB total</span>
            <span>{snap.ollama_process_count} ollama proc</span>
          </div>
        </div>
      )}
    </section>
  );
}
