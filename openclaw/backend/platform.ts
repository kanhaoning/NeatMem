
/**
 * Platform backend — communicates with a NeatMem server over the
 * mem0-compatible REST API (default http://localhost:8790).
 */

import { PLUGIN_VERSION } from "../version.ts";
import {
  APIError,
  type AddOptions,
  AuthError,
  type Backend,
  type DeleteOptions,
  type EntityIds,
  type ListOptions,
  NotFoundError,
  type SearchOptions,
} from "./base.ts";

export class PlatformBackend implements Backend {
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor(config: { apiKey: string; baseUrl: string }) {
    this.baseUrl = config.baseUrl.replace(/\/+$/, "");
    this.headers = {
      Authorization: `Token ${config.apiKey}`,
      "Content-Type": "application/json",
      "X-NeatMem-Source": "OPENCLAW",
      "X-NeatMem-Client-Language": "node",
      "X-NeatMem-Client-Version": PLUGIN_VERSION,
      "X-NeatMem-Caller-Type": "plugin",
    };
  }

  private async _request(
    method: string,
    path: string,
    opts?: { json?: unknown; params?: Record<string, string>; timeout?: number },
  ): Promise<unknown> {
    let url = `${this.baseUrl}${path}`;
    if (opts?.params) {
      const qs = new URLSearchParams(opts.params).toString();
      url += `?${qs}`;
    }

    const fetchOpts: RequestInit = {
      method,
      headers: this.headers,
      signal: AbortSignal.timeout(opts?.timeout ?? 30_000),
    };
    if (opts?.json) {
      fetchOpts.body = JSON.stringify(opts.json);
    }

    const resp = await fetch(url, fetchOpts);

    if (resp.status === 401) {
      throw new AuthError();
    }
    if (resp.status === 404) {
      throw new NotFoundError(path);
    }
    if (resp.status === 400) {
      let detail: string;
      try {
        const body = (await resp.json()) as Record<string, unknown>;
        detail =
          ((body.detail ?? body.message ?? JSON.stringify(body)) as string) ??
          resp.statusText;
      } catch {
        detail = resp.statusText;
      }
      throw new APIError(path, detail);
    }
    if (!resp.ok) {
      let detail: string = resp.statusText;
      try {
        const body = (await resp.json()) as Record<string, unknown>;
        detail = (body.detail ?? body.message ?? resp.statusText) as string;
      } catch {
        /* ignore */
      }
      throw new Error(`HTTP ${resp.status}: ${detail}`);
    }
    if (resp.status === 204) {
      return {};
    }
    return resp.json();
  }

  async add(
    content?: string,
    messages?: Record<string, unknown>[],
    opts: AddOptions = {},
  ): Promise<Record<string, unknown>> {
    const payload: Record<string, unknown> = {};

    if (messages) {
      payload.messages = messages;
    } else if (content) {
      payload.messages = [{ role: "user", content }];
    }

    if (opts.userId) payload.user_id = opts.userId;
    if (opts.agentId) payload.agent_id = opts.agentId;
    if (opts.appId) payload.app_id = opts.appId;
    if (opts.runId) payload.run_id = opts.runId;
    if (opts.metadata) payload.metadata = opts.metadata;
    if (opts.immutable) payload.immutable = true;
    if (opts.infer === false) payload.infer = false;
    if (opts.expires) payload.expiration_date = opts.expires;
    if (opts.categories) payload.categories = opts.categories;
    if (opts.source) payload.source = opts.source;
    if (opts.outputFormat) payload.output_format = opts.outputFormat;
    if (opts.customInstructions)
      payload.custom_instructions = opts.customInstructions;
    if (opts.customCategories)
      payload.custom_categories = opts.customCategories;
    if (opts.deducedMemories) payload.deduced_memories = opts.deducedMemories;

    // Longer timeout: server-side fact extraction (infer) is LLM-bound and
    // can exceed the default 30s on long conversations.
    return (await this._request("POST", "/v1/memories/", {
      json: payload,
      timeout: 120_000,
    })) as Record<string, unknown>;
  }

  private _buildFilters(opts: {
    userId?: string;
    agentId?: string;
    appId?: string;
    runId?: string;
    extraFilters?: Record<string, unknown>;
  }): Record<string, unknown> | undefined {
    // If caller passed a pre-built filter structure, use it directly
    if (
      opts.extraFilters &&
      ("AND" in opts.extraFilters || "OR" in opts.extraFilters)
    ) {
      return opts.extraFilters;
    }

    const andConditions: Record<string, unknown>[] = [];
    if (opts.userId) andConditions.push({ user_id: opts.userId });
    if (opts.agentId) andConditions.push({ agent_id: opts.agentId });
    if (opts.appId) andConditions.push({ app_id: opts.appId });
    if (opts.runId) andConditions.push({ run_id: opts.runId });

    if (opts.extraFilters) {
      for (const [k, v] of Object.entries(opts.extraFilters)) {
        andConditions.push({ [k]: v });
      }
    }

    if (andConditions.length === 1) return andConditions[0];
    if (andConditions.length > 1) return { AND: andConditions };
    return undefined;
  }

  async search(
    query: string,
    opts: SearchOptions = {},
  ): Promise<Record<string, unknown>[]> {
    const payload: Record<string, unknown> = {
      query,
      top_k: opts.topK ?? 10,
      threshold: opts.threshold ?? 0.3,
    };

    // Mirror the mem0 SDK request shape: user_id/run_id appear both
    // top-level and inside filters.
    if (opts.userId) payload.user_id = opts.userId;
    if (opts.runId) payload.run_id = opts.runId;

    if (opts.filters) {
      // Caller-built filter structure (e.g. {AND: [...]}) — use verbatim
      payload.filters = opts.filters;
    } else {
      const apiFilters = this._buildFilters({
        userId: opts.userId,
        agentId: opts.agentId,
        appId: opts.appId,
        runId: opts.runId,
      });
      if (apiFilters) payload.filters = apiFilters;
    }
    if (opts.rerank !== undefined) payload.rerank = opts.rerank;
    if (opts.keyword !== undefined) payload.keyword_search = opts.keyword;
    if (opts.filterMemories !== undefined)
      payload.filter_memories = opts.filterMemories;
    if (opts.categories) payload.categories = opts.categories;
    if (opts.source) payload.source = opts.source;
    if (opts.fields) payload.fields = opts.fields;

    const result = (await this._request("POST", "/v2/memories/search/", {
      json: payload,
    })) as unknown;
    if (Array.isArray(result)) return result;
    const obj = result as Record<string, unknown>;
    return (obj.results ?? obj.memories ?? []) as Record<string, unknown>[];
  }

  async get(memoryId: string): Promise<Record<string, unknown>> {
    return (await this._request("GET", `/v1/memories/${memoryId}/`)) as Record<
      string,
      unknown
    >;
  }

  async listMemories(
    opts: ListOptions = {},
  ): Promise<Record<string, unknown>[]> {
    const payload: Record<string, unknown> = {};

    // Mirror the mem0 SDK request shape: user_id/run_id appear both
    // top-level and inside filters. Pagination params are only sent when
    // explicitly requested — the SDK omits them by default and the server
    // returns its unpaginated default.
    const params: Record<string, string> = {};
    if (opts.page !== undefined) params.page = String(opts.page);
    if (opts.pageSize !== undefined) params.page_size = String(opts.pageSize);

    if (opts.userId) payload.user_id = opts.userId;
    if (opts.runId) payload.run_id = opts.runId;
    if (opts.source) payload.source = opts.source;

    const extra: Record<string, unknown> = {};
    if (opts.category) {
      extra.categories = { contains: opts.category };
    }
    if (opts.after) {
      extra.created_at = {
        ...(extra.created_at as Record<string, unknown> | undefined),
        gte: opts.after,
      };
    }
    if (opts.before) {
      extra.created_at = {
        ...(extra.created_at as Record<string, unknown> | undefined),
        lte: opts.before,
      };
    }

    if (opts.filters) {
      // Caller-built filter structure — use verbatim
      payload.filters = opts.filters;
    } else {
      const apiFilters = this._buildFilters({
        userId: opts.userId,
        agentId: opts.agentId,
        appId: opts.appId,
        runId: opts.runId,
        extraFilters: Object.keys(extra).length > 0 ? extra : undefined,
      });
      if (apiFilters) payload.filters = apiFilters;
    }

    const result = (await this._request("POST", "/v2/memories/", {
      json: payload,
      params: Object.keys(params).length > 0 ? params : undefined,
    })) as unknown;
    if (Array.isArray(result)) return result;
    const obj = result as Record<string, unknown>;
    return (obj.results ?? obj.memories ?? []) as Record<string, unknown>[];
  }

  async update(
    memoryId: string,
    content?: string,
    metadata?: Record<string, unknown>,
  ): Promise<Record<string, unknown>> {
    const payload: Record<string, unknown> = {};
    if (content) payload.text = content;
    if (metadata) payload.metadata = metadata;
    return (await this._request("PUT", `/v1/memories/${memoryId}/`, {
      json: payload,
    })) as Record<string, unknown>;
  }

  async delete(
    memoryId?: string,
    opts: DeleteOptions = {},
  ): Promise<Record<string, unknown>> {
    if (opts.all) {
      const params: Record<string, string> = {};
      if (opts.userId) params.user_id = opts.userId;
      if (opts.agentId) params.agent_id = opts.agentId;
      if (opts.appId) params.app_id = opts.appId;
      if (opts.runId) params.run_id = opts.runId;
      return (await this._request("DELETE", "/v1/memories/", {
        params,
      })) as Record<string, unknown>;
    }
    if (memoryId) {
      return (await this._request(
        "DELETE",
        `/v1/memories/${memoryId}/`,
      )) as Record<string, unknown>;
    }
    throw new Error("Either memoryId or --all is required");
  }

  async deleteEntities(opts: EntityIds): Promise<Record<string, unknown>> {
    // v2 endpoint: DELETE /v2/entities/{entity_type}/{entity_id}/
    const typeMap: [string, string | undefined][] = [
      ["user", opts.userId],
      ["agent", opts.agentId],
      ["app", opts.appId],
      ["run", opts.runId],
    ];
    const entities = typeMap.filter(([, v]) => v) as [string, string][];
    if (entities.length === 0) {
      throw new Error("At least one entity ID is required for deleteEntities.");
    }
    // Delete each provided entity via the v2 path-based endpoint
    let result: Record<string, unknown> = {};
    for (const [entityType, entityId] of entities) {
      result = (await this._request(
        "DELETE",
        `/v2/entities/${entityType}/${entityId}/`,
      )) as Record<string, unknown>;
    }
    return result;
  }

  async ping(): Promise<Record<string, unknown>> {
    return (await this._request("GET", "/v1/ping/")) as Record<string, unknown>;
  }

  async status(
    _opts: { userId?: string; agentId?: string } = {},
  ): Promise<Record<string, unknown>> {
    try {
      await this.ping();
      return { connected: true, backend: "platform", base_url: this.baseUrl };
    } catch (e) {
      return {
        connected: false,
        backend: "platform",
        error: e instanceof Error ? e.message : String(e),
      };
    }
  }

  async entities(entityType: string): Promise<Record<string, unknown>[]> {
    const result = (await this._request("GET", "/v1/entities/")) as unknown;
    let items: Record<string, unknown>[];
    if (Array.isArray(result)) {
      items = result;
    } else {
      items = ((result as Record<string, unknown>).results ?? []) as Record<
        string,
        unknown
      >[];
    }

    const typeMap: Record<string, string> = {
      users: "user",
      agents: "agent",
      apps: "app",
      runs: "run",
    };
    const targetType = typeMap[entityType];
    if (targetType) {
      items = items.filter(
        (e) => (e.type as string | undefined)?.toLowerCase() === targetType,
      );
    }
    return items;
  }

  async listEvents(): Promise<Record<string, unknown>[]> {
    const result = (await this._request("GET", "/v1/events/")) as unknown;
    if (Array.isArray(result)) return result;
    return ((result as Record<string, unknown>).results ?? []) as Record<
      string,
      unknown
    >[];
  }

  async getEvent(eventId: string): Promise<Record<string, unknown>> {
    return (await this._request("GET", `/v1/event/${eventId}/`)) as Record<
      string,
      unknown
    >;
  }

  async history(memoryId: string): Promise<Record<string, unknown>[]> {
    const result = (await this._request(
      "GET",
      `/v1/memories/${memoryId}/history/`,
    )) as unknown;
    if (Array.isArray(result)) return result;
    return ((result as Record<string, unknown>).results ?? []) as Record<
      string,
      unknown
    >[];
  }
}
