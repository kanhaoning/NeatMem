/**
 * Thin REST client for a NeatMem server (mem0-compatible API, default
 * http://localhost:8790). All memory logic lives server-side; this client
 * only carries JSON.
 */

export interface NeatMemClientOptions {
  baseUrl: string;
  /** Optional Token auth; empty means no Authorization header. */
  apiKey: string;
}

export interface SearchOptions {
  userId: string;
  runId?: string;
  agentId?: string;
  topK?: number;
  threshold?: number;
  /** Server-side LLM rerank; auto-recall passes false, tools may pass true. */
  rerank?: boolean;
}

export interface MemoryHit {
  id?: string;
  memory?: string;
  score?: number;
  categories?: string[];
  metadata?: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
  [key: string]: unknown;
}

export class NeatMemError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "NeatMemError";
  }
}

export class NeatMemClient {
  private readonly baseUrl: string;
  private readonly headers: Record<string, string>;

  constructor(options: NeatMemClientOptions) {
    this.baseUrl = options.baseUrl.replace(/\/+$/, "");
    this.headers = {
      "Content-Type": "application/json",
      "X-NeatMem-Source": "DSH",
      "X-NeatMem-Client-Language": "node",
      "X-NeatMem-Caller-Type": "plugin",
    };
    if (options.apiKey) {
      this.headers.Authorization = `Token ${options.apiKey}`;
    }
  }

  private async request(
    method: string,
    path: string,
    opts?: { json?: unknown; timeoutMs?: number; signal?: AbortSignal },
  ): Promise<unknown> {
    const timeout = AbortSignal.timeout(opts?.timeoutMs ?? 30_000);
    const signal = opts?.signal ? AbortSignal.any([opts.signal, timeout]) : timeout;
    let resp: Response;
    try {
      resp = await fetch(`${this.baseUrl}${path}`, {
        method,
        headers: this.headers,
        body: opts?.json === undefined ? undefined : JSON.stringify(opts.json),
        signal,
      });
    } catch (error) {
      throw new NeatMemError(
        `${method} ${path} failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    if (!resp.ok) {
      let detail = resp.statusText;
      try {
        const body = (await resp.json()) as Record<string, unknown>;
        detail = String(body.detail ?? body.message ?? detail);
      } catch {
        /* non-JSON error body — keep statusText */
      }
      throw new NeatMemError(`${method} ${path}: HTTP ${resp.status}: ${detail}`, resp.status);
    }
    if (resp.status === 204) return {};
    return resp.json();
  }

  /** Search memories. Returns the normalized hit list. */
  async search(query: string, opts: SearchOptions): Promise<MemoryHit[]> {
    const payload: Record<string, unknown> = {
      query,
      top_k: opts.topK ?? 10,
      threshold: opts.threshold ?? 0.3,
      user_id: opts.userId,
    };
    if (opts.runId) payload.run_id = opts.runId;
    const conditions: Record<string, unknown>[] = [{ user_id: opts.userId }];
    if (opts.runId) conditions.push({ run_id: opts.runId });
    if (opts.agentId) conditions.push({ agent_id: opts.agentId });
    payload.filters = conditions.length === 1 ? conditions[0] : { AND: conditions };
    if (opts.rerank !== undefined) payload.rerank = opts.rerank;
    const result = await this.request("POST", "/v2/memories/search/", { json: payload });
    return unwrapList(result);
  }

  /** Store-only ingest; the server's scheduler batches extraction. */
  async addMessages(
    messages: Record<string, unknown>[],
    opts: { userId: string; runId?: string; agentId?: string; timeoutMs?: number },
  ): Promise<Record<string, unknown>> {
    const payload: Record<string, unknown> = { messages, user_id: opts.userId };
    if (opts.runId) payload.run_id = opts.runId;
    if (opts.agentId) payload.agent_id = opts.agentId;
    return (await this.request("POST", "/v1/messages/add/", {
      json: payload,
      timeoutMs: opts?.timeoutMs ?? 10_000,
    })) as Record<string, unknown>;
  }

  /** Force synchronous extraction of the pending batch (LLM-bound). */
  async flush(opts: {
    userId: string;
    runId?: string;
    timeoutMs?: number;
    signal?: AbortSignal;
  }): Promise<Record<string, unknown>> {
    const payload: Record<string, unknown> = { user_id: opts.userId };
    if (opts.runId) payload.run_id = opts.runId;
    return (await this.request("POST", "/v1/messages/flush/", {
      json: payload,
      timeoutMs: opts.timeoutMs ?? 60_000,
      signal: opts.signal,
    })) as Record<string, unknown>;
  }

  async get(memoryId: string): Promise<Record<string, unknown>> {
    return (await this.request("GET", `/v1/memories/${memoryId}/`)) as Record<string, unknown>;
  }

  async list(opts: { userId: string; runId?: string }): Promise<MemoryHit[]> {
    // The server's list endpoint only honors `filters` (top-level user_id is
    // ignored), so the scope must live inside the filter structure.
    const conditions: Record<string, unknown>[] = [{ user_id: opts.userId }];
    if (opts.runId) conditions.push({ run_id: opts.runId });
    const payload: Record<string, unknown> = {
      filters: conditions.length === 1 ? conditions[0] : { AND: conditions },
    };
    const result = await this.request("POST", "/v2/memories/", { json: payload });
    return unwrapList(result);
  }

  async update(memoryId: string, text: string): Promise<Record<string, unknown>> {
    return (await this.request("PUT", `/v1/memories/${memoryId}/`, {
      json: { text },
    })) as Record<string, unknown>;
  }

  async delete(memoryId: string): Promise<Record<string, unknown>> {
    return (await this.request("DELETE", `/v1/memories/${memoryId}/`)) as Record<
      string,
      unknown
    >;
  }

  /** Liveness probe used by the fail-open startup warning. */
  async ping(timeoutMs = 3_000): Promise<boolean> {
    try {
      await this.request("GET", "/v1/ping/", { timeoutMs });
      return true;
    } catch {
      return false;
    }
  }
}

function unwrapList(result: unknown): MemoryHit[] {
  if (Array.isArray(result)) return result as MemoryHit[];
  const obj = result as Record<string, unknown> | null;
  return ((obj?.results ?? obj?.memories ?? []) as MemoryHit[]) ?? [];
}
