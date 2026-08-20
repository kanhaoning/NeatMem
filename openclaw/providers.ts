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
import { PlatformBackend } from "./backend/platform.ts";

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

/**
 * Thin adapter over PlatformBackend (hand-written REST client).
 * Replaces the former mem0ai MemoryClient transport — request shapes mirror
 * the mem0 SDK 2.4.5 exactly (verified against tmp/sdk-rest-diff/).
 */
class PlatformProvider implements Mem0Provider {
  private readonly backend: PlatformBackend;

  constructor(apiKey: string, baseUrl?: string) {
    this.backend = new PlatformBackend({
      apiKey,
      baseUrl: baseUrl ?? "http://localhost:8790",
    });
  }

  async add(
    messages: Array<{ role: string; content: string }>,
    options: AddOptions,
  ): Promise<AddResult> {
    const result = await this.backend.add(
      undefined,
      messages as unknown as Record<string, unknown>[],
      {
        userId: options.user_id,
        runId: options.run_id,
        customInstructions: options.custom_instructions,
        customCategories: options.custom_categories,
        outputFormat: options.output_format,
        source: options.source,
        infer: options.infer,
        deducedMemories: options.deduced_memories,
        metadata: options.metadata,
      },
    );
    return normalizeAddResult(result);
  }

  async search(query: string, options: SearchOptions): Promise<MemoryItem[]> {
    // Build the same filter structure the mem0 SDK path sent:
    // flat {user_id, run_id?}, or {AND: [base, callerFilters]} when the
    // caller passed its own filters.
    // NOTE: options.source is intentionally NOT forwarded — the mem0 SDK
    // search payload never included it (verified via tmp/sdk-rest-diff).
    const baseFilters: Record<string, unknown> = { user_id: options.user_id };
    if (options.run_id) baseFilters.run_id = options.run_id;
    const filters = options.filters
      ? { AND: [baseFilters, options.filters] }
      : baseFilters;

    const results = await this.backend.search(query, {
      userId: options.user_id,
      runId: options.run_id,
      topK: options.top_k ?? undefined,
      threshold: options.threshold ?? undefined,
      keyword: options.keyword_search ?? undefined,
      rerank: options.reranking ?? undefined,
      filterMemories: options.filter_memories ?? undefined,
      categories: options.categories ?? undefined,
      filters,
    });
    return normalizeSearchResults(results);
  }

  async get(memoryId: string): Promise<MemoryItem> {
    const result = await this.backend.get(memoryId);
    return normalizeMemoryItem(result);
  }

  async getAll(options: ListOptions): Promise<MemoryItem[]> {
    // NOTE: options.source and options.page_size are intentionally NOT
    // forwarded — the mem0 SDK dropped both from getAll requests
    // (page_size is only honored by the SDK when page is also set).
    const filters: Record<string, unknown> = { user_id: options.user_id };
    if (options.run_id) filters.run_id = options.run_id;

    const results = await this.backend.listMemories({
      userId: options.user_id,
      runId: options.run_id,
      filters,
    });
    return results.map(normalizeMemoryItem);
  }

  async update(memoryId: string, text: string): Promise<void> {
    await this.backend.update(memoryId, text);
  }

  async delete(memoryId: string): Promise<void> {
    await this.backend.delete(memoryId);
  }

  async deleteAll(userId: string): Promise<void> {
    await this.backend.delete(undefined, { all: true, userId });
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
    const result = await this.backend.history(memoryId);
    return result as unknown as Array<{
      id: string;
      old_memory: string;
      new_memory: string;
      event: string;
      created_at: string;
    }>;
  }
}

// ============================================================================
// Provider Factory
// ============================================================================

export function createProvider(cfg: Mem0Config): Mem0Provider {
  return new PlatformProvider(cfg.apiKey!, cfg.baseUrl);
}
