/**
 * Provider implementation: Platform (NeatMem server).
 */

import type {
  Mem0Config,
  Mem0Provider,
  AddOptions,
  SearchOptions,
  ListOptions,
  MemoryItem,
  AddResult,
} from "./types.ts";

// ============================================================================
// Result Normalizers
// ============================================================================

function normalizeMemoryItem(raw: any): MemoryItem {
  return {
    id: raw.id ?? raw.memory_id ?? "",
    memory: raw.memory ?? raw.text ?? raw.content ?? "",
    // Tolerate both snake_case (user_id, created_at) and camelCase field names
    user_id: raw.user_id ?? raw.userId,
    score: raw.score,
    categories: raw.categories,
    metadata: raw.metadata,
    created_at: raw.created_at ?? raw.createdAt,
    updated_at: raw.updated_at ?? raw.updatedAt,
  };
}

function normalizeSearchResults(raw: any): MemoryItem[] {
  // API returns a flat array; tolerate { results: [...] } wrappers
  if (Array.isArray(raw)) return raw.map(normalizeMemoryItem);
  if (raw?.results && Array.isArray(raw.results))
    return raw.results.map(normalizeMemoryItem);
  return [];
}

function normalizeAddResult(raw: any): AddResult {
  // Handle { results: [...] } shape
  if (raw?.results && Array.isArray(raw.results)) {
    return {
      results: raw.results.map((r: any) => ({
        id: r.id ?? r.memory_id ?? "",
        memory: r.memory ?? r.text ?? "",
        // Platform API may return PENDING status (async processing)
        event:
          r.event ??
          r.metadata?.event ??
          (r.status === "PENDING" ? "ADD" : "ADD"),
      })),
    };
  }
  // Platform API without output_format returns flat array
  if (Array.isArray(raw)) {
    return {
      results: raw.map((r: any) => ({
        id: r.id ?? r.memory_id ?? "",
        memory: r.memory ?? r.text ?? "",
        event:
          r.event ??
          r.metadata?.event ??
          (r.status === "PENDING" ? "ADD" : "ADD"),
      })),
    };
  }
  return { results: [] };
}

// ============================================================================
// Platform Provider (NeatMem server)
// ============================================================================

class PlatformProvider implements Mem0Provider {
  private client: any; // MemoryClient from mem0ai
  private initPromise: Promise<void> | null = null;

  constructor(
    private readonly apiKey: string,
    private readonly baseUrl?: string,
  ) {}

  private async ensureClient(): Promise<void> {
    if (this.client) return;
    if (this.initPromise) return this.initPromise;
    this.initPromise = this._init().catch((err) => {
      this.initPromise = null;
      throw err;
    });
    return this.initPromise;
  }

  private async _init(): Promise<void> {
    const { default: MemoryClient } = await import("mem0ai");
    const opts: {
      apiKey: string;
      host?: string;
    } = {
      apiKey: this.apiKey,
    };
    if (this.baseUrl) opts.host = this.baseUrl;
    this.client = new MemoryClient(opts);
  }

  async add(
    messages: Array<{ role: string; content: string }>,
    options: AddOptions,
  ): Promise<AddResult> {
    await this.ensureClient();
    const opts: Record<string, unknown> = { user_id: options.user_id };
    if (options.run_id) opts.run_id = options.run_id;
    if (options.custom_instructions)
      opts.custom_instructions = options.custom_instructions;
    if (options.custom_categories)
      opts.custom_categories = options.custom_categories;
    if (options.output_format) opts.output_format = options.output_format;
    if (options.source) opts.source = options.source;
    // Agentic harness: direct storage bypass
    if (options.infer !== undefined) opts.infer = options.infer;
    if (options.deduced_memories)
      opts.deduced_memories = options.deduced_memories;
    if (options.metadata) opts.metadata = options.metadata;
    if (options.expiration_date) opts.expiration_date = options.expiration_date;
    if (options.immutable) opts.immutable = options.immutable;

    const result = await this.client.add(messages, opts);
    return normalizeAddResult(result);
  }

  async search(query: string, options: SearchOptions): Promise<MemoryItem[]> {
    await this.ensureClient();
    const opts: Record<string, unknown> = {
      api_version: "v2",
      user_id: options.user_id,
    };
    if (options.run_id) opts.run_id = options.run_id;
    if (options.top_k != null) opts.top_k = options.top_k;
    if (options.threshold != null) opts.threshold = options.threshold;
    if (options.keyword_search != null)
      opts.keyword_search = options.keyword_search;
    if (options.reranking != null) opts.rerank = options.reranking;
    if (options.filter_memories != null)
      opts.filter_memories = options.filter_memories;
    if (options.categories != null) opts.categories = options.categories;
    const baseFilters: Record<string, unknown> = { user_id: options.user_id };
    if (options.run_id) baseFilters.run_id = options.run_id;

    if (options.filters) {
      opts.filters = { AND: [baseFilters, options.filters] };
    } else {
      opts.filters = baseFilters;
    }

    const results = await this.client.search(query, opts);
    return normalizeSearchResults(results);
  }

  async get(memoryId: string): Promise<MemoryItem> {
    await this.ensureClient();
    const result = await this.client.get(memoryId);
    return normalizeMemoryItem(result);
  }

  async getAll(options: ListOptions): Promise<MemoryItem[]> {
    await this.ensureClient();
    const opts: Record<string, unknown> = {
      api_version: "v2",
      user_id: options.user_id,
      filters: { user_id: options.user_id },
    };
    if (options.run_id) {
      opts.run_id = options.run_id;
      (opts.filters as Record<string, unknown>).run_id = options.run_id;
    }
    if (options.page_size != null) opts.page_size = options.page_size;

    const results = await this.client.getAll(opts);
    if (Array.isArray(results)) return results.map(normalizeMemoryItem);
    // Some versions return { results: [...] }
    if (results?.results && Array.isArray(results.results))
      return results.results.map(normalizeMemoryItem);
    return [];
  }

  async update(memoryId: string, text: string): Promise<void> {
    await this.ensureClient();
    await this.client.update(memoryId, { text });
  }

  async delete(memoryId: string): Promise<void> {
    await this.ensureClient();
    await this.client.delete(memoryId);
  }

  async deleteAll(userId: string): Promise<void> {
    await this.ensureClient();
    await this.client.deleteAll({ user_id: userId });
  }

  async history(memoryId: string): Promise<
    Array<{
      id: string;
      old_memory: string;
      new_memory: string;
      event: string;
      created_at: string;
    }>
  > {
    await this.ensureClient();
    const result = await this.client.history(memoryId);
    return Array.isArray(result) ? result : [];
  }
}

// ============================================================================
// Provider Factory
// ============================================================================

export function createProvider(cfg: Mem0Config): Mem0Provider {
  return new PlatformProvider(cfg.apiKey!, cfg.baseUrl);
}
