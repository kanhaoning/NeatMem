/**
 * Abstract backend interface and error classes.
 */

export interface AddOptions {
  userId?: string;
  agentId?: string;
  appId?: string;
  runId?: string;
  metadata?: Record<string, unknown>;
  immutable?: boolean;
  infer?: boolean;
  expires?: string;
  categories?: string[];
  // Passthrough fields used by the provider adapter (mem0 protocol)
  source?: string;
  outputFormat?: string;
  customInstructions?: string;
  customCategories?: Array<Record<string, string>>;
  deducedMemories?: string[];
}

export interface SearchOptions {
  userId?: string;
  agentId?: string;
  appId?: string;
  runId?: string;
  topK?: number;
  threshold?: number;
  rerank?: boolean;
  keyword?: boolean;
  // filters is used verbatim when provided (skips built filter assembly)
  filters?: Record<string, unknown>;
  fields?: string[];
  // Passthrough fields used by the provider adapter (mem0 protocol)
  filterMemories?: boolean;
  categories?: string[];
  source?: string;
}

export interface ListOptions {
  userId?: string;
  agentId?: string;
  appId?: string;
  runId?: string;
  page?: number;
  pageSize?: number;
  category?: string;
  after?: string;
  before?: string;
  // Passthrough fields used by the provider adapter (mem0 protocol).
  // filters is used verbatim when provided (skips built filter assembly).
  source?: string;
  filters?: Record<string, unknown>;
}

export interface DeleteOptions {
  all?: boolean;
  userId?: string;
  agentId?: string;
  appId?: string;
  runId?: string;
}

export interface EntityIds {
  userId?: string;
  agentId?: string;
  appId?: string;
  runId?: string;
}

export interface Backend {
  add(
    content?: string,
    messages?: Record<string, unknown>[],
    opts?: AddOptions,
  ): Promise<Record<string, unknown>>;

  search(
    query: string,
    opts?: SearchOptions,
  ): Promise<Record<string, unknown>[]>;

  get(memoryId: string): Promise<Record<string, unknown>>;

  listMemories(opts?: ListOptions): Promise<Record<string, unknown>[]>;

  update(
    memoryId: string,
    content?: string,
    metadata?: Record<string, unknown>,
  ): Promise<Record<string, unknown>>;

  delete(
    memoryId?: string,
    opts?: DeleteOptions,
  ): Promise<Record<string, unknown>>;

  deleteEntities(opts: EntityIds): Promise<Record<string, unknown>>;

  status(opts?: {
    userId?: string;
    agentId?: string;
  }): Promise<Record<string, unknown>>;

  entities(entityType: string): Promise<Record<string, unknown>[]>;

  listEvents(): Promise<Record<string, unknown>[]>;

  getEvent(eventId: string): Promise<Record<string, unknown>>;

  history(memoryId: string): Promise<Record<string, unknown>[]>;
}

export class AuthError extends Error {
  constructor(
    message = "Authentication failed. Your API key may be invalid or expired.",
  ) {
    super(message);
    this.name = "AuthError";
  }
}

export class NotFoundError extends Error {
  constructor(path: string) {
    super(`Resource not found: ${path}`);
    this.name = "NotFoundError";
  }
}

export class APIError extends Error {
  constructor(path: string, detail: string) {
    super(`Bad request to ${path}: ${detail}`);
    this.name = "APIError";
  }
}
