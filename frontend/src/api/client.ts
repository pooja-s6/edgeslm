/** Typed FastAPI client for EdgeSLM Lab. */

export type TaskType =
  | "classification"
  | "extraction"
  | "summarization"
  | "reasoning";

export interface HealthResponse {
  status: string;
  ollama_reachable: boolean;
  detail: string | null;
}

export interface HardwareSnapshot {
  cpu_percent: number;
  total_ram_mb: number;
  available_ram_mb: number;
  used_ram_mb: number;
  runner_rss_mb: number;
  ollama_process_count: number;
}

export interface ModelInfo {
  ollama_tag: string;
  family: string;
  params_b: number;
  quantization: string;
  estimated_ram_mb: number;
  complexity_tier: string;
  installed_locally: boolean;
}

export interface RoutingDecisionOut {
  selected_model: string;
  quantization: string;
  complexity_score: number;
  hardware_constrained: boolean;
  rationale: string;
  available_ram_mb: number;
}

export interface InferenceTelemetryOut {
  load_time_ms: number;
  ttft_ms: number;
  decode_time_ms: number;
  tokens_per_sec: number;
  total_time_ms: number;
  eval_count: number;
  runner_rss_before_mb: number | null;
  runner_rss_peak_mb: number | null;
  available_ram_mb: number | null;
  cpu_percent: number | null;
}

export interface AdaptiveInferResponse {
  routing: RoutingDecisionOut;
  response: string;
  telemetry: InferenceTelemetryOut;
}

export interface BenchmarkResultRow {
  id: number;
  suite_item_id: string;
  task_type: string;
  model_tag: string;
  quantization: string;
  quality_score: number;
  load_time_ms: number;
  ttft_ms: number;
  decode_time_ms: number;
  tokens_per_sec: number;
  total_time_ms: number;
  runner_rss_before_mb: number | null;
  runner_rss_peak_mb: number | null;
  available_ram_mb: number | null;
  cpu_percent: number | null;
  created_at: string | null;
}

export interface BenchmarkResultsResponse {
  count: number;
  results: BenchmarkResultRow[];
}

export interface BenchmarkRunResponse {
  runs_completed: number;
  cooldown_seconds: number;
  results: Array<{
    suite_item_id: string;
    task_type: string;
    model_tag: string;
    quantization: string;
    quality_score: number;
    load_time_ms: number;
    ttft_ms: number;
    decode_time_ms: number;
    tokens_per_sec: number;
    total_time_ms: number;
    runner_rss_before_mb: number | null;
    runner_rss_peak_mb: number | null;
  }>;
}

const BASE = "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/api/health"),
  hardware: () => request<HardwareSnapshot>("/api/hardware"),
  models: () =>
    request<{ models: ModelInfo[] }>("/api/models").then((r) => r.models),
  inferAdaptive: (body: {
    prompt: string;
    task_type?: TaskType | null;
  }) =>
    request<AdaptiveInferResponse>("/api/infer/adaptive", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  benchmarkResults: (limit = 200) =>
    request<BenchmarkResultsResponse>(`/api/benchmarks/results?limit=${limit}`),
  runBenchmarks: (body: {
    model_tags?: string[] | null;
    use_adaptive_routing?: boolean;
    cooldown_seconds?: number;
    limit?: number | null;
  }) => {
    const payload: Record<string, unknown> = {
      cooldown_seconds: body.cooldown_seconds ?? 3,
      use_adaptive_routing: body.use_adaptive_routing ?? false,
    };
    if (body.model_tags != null) payload.model_tags = body.model_tags;
    if (body.limit != null) payload.limit = body.limit;
    return request<BenchmarkRunResponse>("/api/benchmarks/run", {
      method: "POST",
      body: JSON.stringify(payload),
    });
  },
  exportCsvUrl: (limit = 10000) => `/api/export/csv?limit=${limit}`,
  exportJsonUrl: (limit = 10000) => `/api/export/json?limit=${limit}`,
};
