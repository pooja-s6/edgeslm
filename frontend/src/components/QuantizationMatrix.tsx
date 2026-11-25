import { useEffect, useMemo, useState } from "react";
import {
  api,
  type BenchmarkResultRow,
  type ModelInfo,
} from "../api/client";

interface AggregateRow {
  model_tag: string;
  quantization: string;
  family: string;
  estimated_ram_mb: number;
  installed: boolean;
  n: number;
  avg_quality: number | null;
  avg_ttft_ms: number | null;
  avg_tok_s: number | null;
  avg_rss_mb: number | null;
}

function avg(nums: number[]): number | null {
  if (!nums.length) return null;
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function fmt(n: number | null, digits = 1): string {
  if (n === null || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

interface Props {
  results: BenchmarkResultRow[];
  onRefreshResults: () => void;
}

export default function QuantizationMatrix({
  results,
  onRefreshResults,
}: Props) {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    void api.models().then(setModels).catch(() => setModels([]));
  }, []);

  const installedTags = useMemo(
    () => models.filter((m) => m.installed_locally).map((m) => m.ollama_tag),
    [models],
  );

  const rows: AggregateRow[] = useMemo(() => {
    return models.map((m) => {
      const subset = results.filter((r) => r.model_tag === m.ollama_tag);
      return {
        model_tag: m.ollama_tag,
        quantization: m.quantization,
        family: m.family,
        estimated_ram_mb: m.estimated_ram_mb,
        installed: m.installed_locally,
        n: subset.length,
        avg_quality: avg(subset.map((r) => r.quality_score)),
        avg_ttft_ms: avg(subset.map((r) => r.ttft_ms)),
        avg_tok_s: avg(subset.map((r) => r.tokens_per_sec)),
        avg_rss_mb: avg(
          subset
            .map((r) => r.runner_rss_peak_mb)
            .filter((v): v is number => v != null),
        ),
      };
    });
  }, [models, results]);

  async function runSweep(mode: "smoke" | "full") {
    setRunning(true);
    setMessage(null);
    try {
      const tags = installedTags.length
        ? installedTags
        : models.slice(0, 1).map((m) => m.ollama_tag);
      const res = await api.runBenchmarks({
        model_tags: mode === "smoke" ? tags.slice(0, 1) : tags,
        cooldown_seconds: mode === "smoke" ? 1 : 2,
        limit: mode === "smoke" ? 2 : null,
      });
      setMessage(
        mode === "smoke"
          ? `Smoke complete: ${res.runs_completed} runs`
          : `Full sweep complete: ${res.runs_completed} runs across ${tags.length} models`,
      );
      onRefreshResults();
      void api.models().then(setModels).catch(() => undefined);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Benchmark failed");
    } finally {
      setRunning(false);
    }
  }

  return (
    <section className="panel">
      <header className="panel-header panel-header--row">
        <div>
          <div className="panel-kicker">Experiment matrix</div>
          <h2>Quantization Matrix</h2>
          <p>
            Compare estimated footprint, TTFT, throughput, runner RSS, and task
            quality across installed Q4 / Q8 variants.
          </p>
        </div>
        <div className="action-stack">
          <button
            className="btn btn--ghost"
            type="button"
            onClick={() => void runSweep("smoke")}
            disabled={running}
          >
            {running ? "Running…" : "Smoke (2 items)"}
          </button>
          <button
            className="btn"
            type="button"
            onClick={() => void runSweep("full")}
            disabled={running || installedTags.length === 0}
          >
            {running ? "Sweep in progress…" : "Run full suite"}
          </button>
        </div>
      </header>

      {message && <p className="status">{message}</p>}

      <div className="table-wrap">
        <table className="matrix">
          <thead>
            <tr>
              <th>Model</th>
              <th>Quant</th>
              <th>Est. RAM</th>
              <th>Local</th>
              <th>N</th>
              <th>Quality</th>
              <th>TTFT ms</th>
              <th>Tok/s</th>
              <th>RSS MB</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.model_tag} className={r.installed ? undefined : "row-muted"}>
                <td className="mono">{r.model_tag}</td>
                <td>
                  <span
                    className={`chip chip--${r.quantization === "Q8_0" ? "q8" : "q4"}`}
                  >
                    {r.quantization}
                  </span>
                </td>
                <td className="mono">{r.estimated_ram_mb}</td>
                <td>{r.installed ? "yes" : "no"}</td>
                <td className="mono">{r.n}</td>
                <td className="mono">{fmt(r.avg_quality, 3)}</td>
                <td className="mono">{fmt(r.avg_ttft_ms, 0)}</td>
                <td className="mono">{fmt(r.avg_tok_s, 1)}</td>
                <td className="mono">{fmt(r.avg_rss_mb, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
