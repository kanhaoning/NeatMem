/**
 * Shared type definitions for the OpenClaw NeatMem plugin.
 */

export type Mem0Config = {
  apiKey?: string;
  baseUrl?: string;
  customInstructions: string;
  customCategories: Record<string, string>;
  // Shared
  userId: string;
  autoCapture: boolean;
  autoRecall: boolean;
  searchThreshold: number;
  topK: number;
  // Setup state
  needsSetup?: boolean;
  // Recall tuning
  recall?: RecallConfig;
};

export interface AddOptions {
  user_id: string;
  run_id?: string;
  custom_instructions?: string;
  custom_categories?: Array<Record<string, string>>;
  output_format?: string;
  source?: string;
  // Direct-store additions (hidden memory_add tool)
  infer?: boolean;
  deduced_memories?: string[];
  metadata?: Record<string, unknown>;
}

export interface SearchOptions {
  user_id: string;
  run_id?: string;
  top_k?: number;
  threshold?: number;
  limit?: number;
  keyword_search?: boolean;
  reranking?: boolean;
  filter_memories?: boolean;
  categories?: string[];
  filters?: Record<string, unknown>;
  source?: string;
}

// ============================================================================
// Recall Configuration
// ============================================================================

export interface RecallConfig {
  rerank?: boolean;
  keywordSearch?: boolean;
  filterMemories?: boolean;
  threshold?: number;
}

export interface ListOptions {
  user_id: string;
  run_id?: string;
  page_size?: number;
  source?: string;
}

export interface MemoryItem {
  id: string;
  memory: string;
  user_id?: string;
  score?: number;
  categories?: string[];
  metadata?: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
}

export interface AddResultItem {
  id: string;
  memory: string;
  event: "ADD" | "UPDATE" | "DELETE" | "NOOP";
}

export interface AddResult {
  results: AddResultItem[];
}

export interface Mem0Provider {
  add(
    messages: Array<{ role: string; content: string }>,
    options: AddOptions,
  ): Promise<AddResult>;
  search(query: string, options: SearchOptions): Promise<MemoryItem[]>;
  get(memoryId: string): Promise<MemoryItem>;
  getAll(options: ListOptions): Promise<MemoryItem[]>;
  update(memoryId: string, text: string): Promise<void>;
  delete(memoryId: string): Promise<void>;
  deleteAll(userId: string): Promise<void>;
  history(
    memoryId: string,
  ): Promise<
    Array<{
      id: string;
      old_memory: string;
      new_memory: string;
      event: string;
      created_at: string;
    }>
  >;
}
