import { useCallback, useEffect, useState } from "react";
import { api, type BenchmarkResultRow, type HealthResponse } from "./api/client";
import HardwareMonitor from "./components/HardwareMonitor";
import LiveRouterPlayground from "./components/LiveRouterPlayground";
import ParetoChart from "./components/ParetoChart";
import QuantizationMatrix from "./components/QuantizationMatrix";

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [results, setResults] = useState<BenchmarkResultRow[]>([]);

  const refreshResults = useCallback(() => {
    void api
      .benchmarkResults(1000)
      .then((r) => setResults(r.results))
      .catch(() => setResults([]));
  }, []);

  useEffect(() => {
    void api
      .health()
      .then(setHealth)
      .catch(() =>
        setHealth({
          status: "unreachable",
          ollama_reachable: false,
          detail: "Backend not reachable on :8000",
        }),
      );
    refreshResults();
    const id = window.setInterval(refreshResults, 15000);
    return () => window.clearInterval(id);
  }, [refreshResults]);

  return (
    <div className="app-shell">
      <div className="atmosphere" aria-hidden />

      <header className="topbar">
        <div className="brand-block">
          <p className="eyebrow">Edge systems research</p>
          <h1 className="brand">EdgeSLM Lab</h1>
        </div>
        <div className="hero-status">
          <span
            className={`pill ${health?.status === "ok" ? "pill--ok" : "pill--warn"}`}
          >
            API {health?.status ?? "…"}
          </span>
          <span
            className={`pill ${health?.ollama_reachable ? "pill--ok" : "pill--warn"}`}
          >
            Ollama {health?.ollama_reachable ? "online" : "offline"}
          </span>
          <span className="pill pill--muted mono">{results.length} stored runs</span>
          <a className="pill pill--link" href={api.exportCsvUrl()}>
            Export CSV
          </a>
          <a className="pill pill--link" href={api.exportJsonUrl()}>
            Export JSON
          </a>
        </div>
      </header>

      <section className="intro">
        <h2 className="intro-title">
          Resource-aware SLM benchmarking on CPU-constrained hardware
        </h2>
        <p className="lede">
          Evaluate quantized Small Language Models with real Ollama telemetry,
          then route prompts by complexity and live memory budget—without a GPU.
        </p>
      </section>

      <main className="workspace">
        <div className="workspace-primary">
          <LiveRouterPlayground />
          <QuantizationMatrix
            results={results}
            onRefreshResults={refreshResults}
          />
          <ParetoChart results={results} />
        </div>
        <aside className="workspace-side">
          <HardwareMonitor />
        </aside>
      </main>

      <footer className="site-footer">
        <span>EdgeSLM Lab</span>
        <span className="mono">Quality · Latency · Memory</span>
      </footer>
    </div>
  );
}
