import { useState, type FormEvent } from "react";
import {
  api,
  type AdaptiveInferResponse,
  type TaskType,
} from "../api/client";

const TASKS: Array<{ value: TaskType | ""; label: string }> = [
  { value: "", label: "Auto-detect" },
  { value: "classification", label: "Classification" },
  { value: "extraction", label: "JSON extraction" },
  { value: "summarization", label: "Summarization" },
  { value: "reasoning", label: "Multi-step reasoning" },
];

const SAMPLES: Array<{
  id: string;
  label: string;
  task_type: TaskType;
  prompt: string;
}> = [
  {
    id: "simple",
    label: "Simple · 1B-Q4",
    task_type: "classification",
    prompt:
      "Classify the sentiment of this review as Positive or Negative: The delivery was fast and the packaging was great.",
  },
  {
    id: "medium",
    label: "Medium · 1.5B / 1B-Q8",
    task_type: "extraction",
    prompt:
      "Extract the company name, role, and location as valid JSON from: Hexaware is hiring a Backend Intern in Chennai.",
  },
  {
    id: "hard",
    label: "Hard · 3B-Q4",
    task_type: "reasoning",
    prompt:
      "A server processes 45 requests per second at Q4 precision and 28 requests per second at Q8. If a batch of 1,000 requests splits 60% to Q4 and 40% to Q8, what is the total processing time? Show step-by-step math.",
  },
];

export default function LiveRouterPlayground() {
  const [prompt, setPrompt] = useState(SAMPLES[0].prompt);
  const [taskType, setTaskType] = useState<TaskType | "">(SAMPLES[0].task_type);
  const [activeSample, setActiveSample] = useState("simple");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AdaptiveInferResponse | null>(null);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const data = await api.inferAdaptive({
        prompt,
        task_type: taskType || null,
      });
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inference failed");
    } finally {
      setLoading(false);
    }
  }

  function loadSample(id: string) {
    const sample = SAMPLES.find((s) => s.id === id);
    if (!sample) return;
    setActiveSample(id);
    setPrompt(sample.prompt);
    setTaskType(sample.task_type);
    setResult(null);
    setError(null);
  }

  return (
    <section className="panel">
      <header className="panel-header">
        <div className="panel-kicker">Adaptive inference</div>
        <h2>Live Router Playground</h2>
        <p>
          Submit a prompt to score complexity, select a model-quantization pair,
          and capture real Ollama timing and memory telemetry.
        </p>
      </header>

      <div className="sample-row" role="group" aria-label="Difficulty samples">
        {SAMPLES.map((s) => (
          <button
            key={s.id}
            type="button"
            className={
              activeSample === s.id ? "sample-btn sample-btn--active" : "sample-btn"
            }
            onClick={() => loadSample(s.id)}
          >
            {s.label}
          </button>
        ))}
      </div>

      <form className="router-form" onSubmit={onSubmit}>
        <label className="field">
          <span>Task type</span>
          <select
            value={taskType}
            onChange={(e) => setTaskType(e.target.value as TaskType | "")}
          >
            {TASKS.map((t) => (
              <option key={t.label} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span>Prompt</span>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={4}
            required
          />
        </label>

        <button className="btn" type="submit" disabled={loading || !prompt.trim()}>
          {loading ? "Routing and generating…" : "Run adaptive inference"}
        </button>
      </form>

      {error && <p className="status status--warn">{error}</p>}

      {result && (
        <div className="trace">
          <div className="trace-grid">
            <div>
              <span className="label">Complexity</span>
              <strong className="mono">
                {result.routing.complexity_score.toFixed(3)}
              </strong>
            </div>
            <div>
              <span className="label">Selected SLM</span>
              <strong className="mono">{result.routing.selected_model}</strong>
            </div>
            <div>
              <span className="label">Quantization</span>
              <strong className="mono">{result.routing.quantization}</strong>
            </div>
            <div>
              <span className="label">Hardware gate</span>
              <strong
                className={
                  result.routing.hardware_constrained ? "tone-warn" : "tone-ok"
                }
              >
                {result.routing.hardware_constrained
                  ? "Constrained"
                  : "Within budget"}
              </strong>
            </div>
          </div>

          <p className="rationale">{result.routing.rationale}</p>

          <div className="telemetry-row mono">
            <span>TTFT {result.telemetry.ttft_ms.toFixed(0)} ms</span>
            <span>{result.telemetry.tokens_per_sec.toFixed(1)} tok/s</span>
            <span>Total {result.telemetry.total_time_ms.toFixed(0)} ms</span>
            <span>
              RSS peak{" "}
              {result.telemetry.runner_rss_peak_mb?.toFixed(0) ?? "—"} MB
            </span>
          </div>

          <pre className="response-block">{result.response}</pre>
        </div>
      )}
    </section>
  );
}
