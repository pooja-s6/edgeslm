import { useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { BenchmarkResultRow } from "../api/client";

type AxisMode = "latency" | "ram";

interface Point {
  x: number;
  y: number;
  z: number;
  model: string;
  quant: string;
  task: string;
}

function familyColor(tag: string): string {
  if (tag.includes("llama")) return "#0f766e";
  if (tag.includes("1.5b")) return "#b45309";
  if (tag.includes("3b")) return "#1d4ed8";
  return "#475569";
}

interface Props {
  results: BenchmarkResultRow[];
}

export default function ParetoChart({ results }: Props) {
  const [mode, setMode] = useState<AxisMode>("latency");

  const series = useMemo(() => {
    const byModel = new Map<string, Point[]>();
    for (const r of results) {
      const x =
        mode === "latency"
          ? r.total_time_ms
          : (r.runner_rss_peak_mb ?? r.available_ram_mb ?? 0);
      if (!Number.isFinite(x) || x <= 0) continue;
      const point: Point = {
        x,
        y: r.quality_score,
        z: Math.max(8, r.tokens_per_sec),
        model: r.model_tag,
        quant: r.quantization,
        task: r.task_type,
      };
      const list = byModel.get(r.model_tag) ?? [];
      list.push(point);
      byModel.set(r.model_tag, list);
    }
    return [...byModel.entries()].map(([model, data]) => ({ model, data }));
  }, [results, mode]);

  return (
    <section className="panel">
      <header className="panel-header panel-header--row">
        <div>
          <div className="panel-kicker">Trade-off analysis</div>
          <h2>Pareto Frontier</h2>
          <p>
            Quality versus {mode === "latency" ? "end-to-end latency" : "runner RAM"}{" "}
            from persisted benchmark runs
          </p>
        </div>
        <div className="seg">
          <button
            type="button"
            className={mode === "latency" ? "seg-btn active" : "seg-btn"}
            onClick={() => setMode("latency")}
          >
            Latency
          </button>
          <button
            type="button"
            className={mode === "ram" ? "seg-btn active" : "seg-btn"}
            onClick={() => setMode("ram")}
          >
            RAM
          </button>
        </div>
      </header>

      {results.length === 0 ? (
        <p className="status">
          No benchmark rows yet. Run a smoke benchmark or full suite to populate the
          frontier.
        </p>
      ) : (
        <div className="chart-frame">
          <ResponsiveContainer width="100%" height={340}>
            <ScatterChart margin={{ top: 12, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid stroke="rgba(15, 23, 42, 0.08)" />
              <XAxis
                type="number"
                dataKey="x"
                name={mode === "latency" ? "Total latency" : "RSS"}
                unit={mode === "latency" ? " ms" : " MB"}
                tick={{ fill: "#64748b", fontSize: 12 }}
              />
              <YAxis
                type="number"
                dataKey="y"
                name="Quality"
                domain={[0, 1]}
                tick={{ fill: "#64748b", fontSize: 12 }}
              />
              <ZAxis type="number" dataKey="z" range={[40, 160]} />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                contentStyle={{
                  background: "#0f172a",
                  border: "none",
                  borderRadius: 8,
                  color: "#f8fafc",
                  fontFamily: "IBM Plex Mono, monospace",
                  fontSize: 12,
                }}
                formatter={(value: number, name: string) => [
                  typeof value === "number" ? value.toFixed(3) : value,
                  name,
                ]}
                labelFormatter={() => ""}
              />
              <Legend />
              {series.map(({ model, data }) => (
                <Scatter
                  key={model}
                  name={model.replace(/-instruct/, "")}
                  data={data}
                  fill={familyColor(model)}
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}
