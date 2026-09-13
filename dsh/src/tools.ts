/**
 * The five `memory_*` tools exposed to the agent. `memory_search` is the
 * accuracy path (rerank exposed to the model); the rest are thin CRUD over
 * the NeatMem REST API. All results are bounded by `maxBodyChars`.
 */

import type { Context } from "@deepseek-ai/cordis";
import { defineTool } from "@deepseek-ai/dsh-tools";

import type { MemoryHit, NeatMemClient } from "./client.js";

export interface MemoryToolsOptions {
  client: NeatMemClient;
  userId: string;
  agentId: string;
  maxBodyChars: number;
  onWarn: (message: string, error?: unknown) => void;
}

const JSON_OUTPUT = {
  schema: {
    type: "object" as const,
    additionalProperties: true,
    properties: {
      text: { type: "string" as const, required: true },
    },
  },
  render: (_args: unknown, value: Record<string, unknown>) => [
    {
      type: "text" as const,
      text: typeof value["text"] === "string" ? value["text"] : JSON.stringify(value),
    },
  ],
} as const;

function clip(text: string, maxChars: number): string {
  return text.length <= maxChars ? text : `${text.slice(0, maxChars - 1)}…`;
}

function formatHits(hits: MemoryHit[], maxBodyChars: number): string {
  if (hits.length === 0) return "No memories found.";
  const lines = hits.map((hit, index) => {
    const score = typeof hit.score === "number" ? hit.score.toFixed(3) : "?";
    return `${index + 1}. [id=${hit.id ?? "?"} score=${score}] ${String(hit.memory ?? "")}`;
  });
  return clip(lines.join("\n"), maxBodyChars);
}

/** Current dsh session id for `sessionScope`, when the loop provides an agent. */
function sessionIdOf(agent: unknown): string | undefined {
  const session = (agent as { session?: { id?: unknown } } | undefined)?.session;
  return typeof session?.id === "string" ? session.id : undefined;
}

export function registerMemoryTools(
  ctx: Context,
  options: MemoryToolsOptions,
): () => void {
  const disposers: Array<() => void> = [];
  const { client } = options;

  disposers.push(ctx.tools.register(defineTool({
    name: "memory_search",
    description:
      "Search long-term memory for facts about the user, their preferences, projects, and prior " +
      "decisions. Use this before claiming that earlier user context is unavailable. Set rerank " +
      "to true for higher accuracy on ambiguous queries at the cost of latency.",
    parameters: {
      query: {
        type: "string",
        required: true,
        description: "A concise free-text memory query.",
      },
      topK: {
        type: "integer",
        description: "Maximum results (1-50, default 10).",
      },
      threshold: {
        type: "number",
        description: "Minimum relevance score (0-1, default 0.3).",
      },
      rerank: {
        type: "boolean",
        description: "Enable server-side LLM reranking (slower, more accurate). Default false.",
      },
      sessionScope: {
        type: "boolean",
        description: "Restrict results to memories captured in the current session.",
      },
    },
    output: JSON_OUTPUT,
    isConcurrencySafe: () => true,
    async execute(args, exec) {
      const query = String(args.query ?? "").trim();
      if (!query) throw new Error("memory_search: `query` is required");
      const topK = Math.min(Math.max(Number(args.topK) || 10, 1), 50);
      const runId = args.sessionScope === true ? sessionIdOf(exec.agent) : undefined;
      const hits = await client.search(query, {
        userId: options.userId,
        agentId: options.agentId || undefined,
        runId,
        topK,
        threshold: typeof args.threshold === "number" ? args.threshold : 0.3,
        rerank: args.rerank === true,
      });
      return { text: formatHits(hits, options.maxBodyChars), count: hits.length };
    },
  })));

  disposers.push(ctx.tools.register(defineTool({
    name: "memory_get",
    description: "Fetch one memory item by its id.",
    parameters: {
      id: { type: "string", required: true, description: "Memory item id." },
    },
    output: JSON_OUTPUT,
    isConcurrencySafe: () => true,
    async execute(args) {
      const id = String(args.id ?? "").trim();
      if (!id) throw new Error("memory_get: `id` is required");
      const memory = await client.get(id);
      return { text: clip(JSON.stringify(memory, null, 2), options.maxBodyChars) };
    },
  })));

  disposers.push(ctx.tools.register(defineTool({
    name: "memory_list",
    description: "List stored memories for the current user namespace.",
    parameters: {
      sessionScope: {
        type: "boolean",
        description: "Restrict to memories captured in the current session.",
      },
    },
    output: JSON_OUTPUT,
    isConcurrencySafe: () => true,
    async execute(args, exec) {
      const runId = args.sessionScope === true ? sessionIdOf(exec.agent) : undefined;
      const hits = await client.list({ userId: options.userId, runId });
      return { text: formatHits(hits, options.maxBodyChars), count: hits.length };
    },
  })));

  disposers.push(ctx.tools.register(defineTool({
    name: "memory_update",
    description: "Replace the text of one stored memory by id.",
    parameters: {
      id: { type: "string", required: true, description: "Memory item id." },
      text: { type: "string", required: true, description: "Replacement memory text." },
    },
    output: JSON_OUTPUT,
    isConcurrencySafe: () => false,
    async execute(args) {
      const id = String(args.id ?? "").trim();
      const text = String(args.text ?? "").trim();
      if (!id || !text) throw new Error("memory_update: `id` and `text` are required");
      await client.update(id, text);
      return { text: `Memory ${id} updated.` };
    },
  })));

  disposers.push(ctx.tools.register(defineTool({
    name: "memory_delete",
    description: "Delete one stored memory by id. This is irreversible.",
    parameters: {
      id: { type: "string", required: true, description: "Memory item id." },
    },
    output: JSON_OUTPUT,
    isConcurrencySafe: () => false,
    async execute(args) {
      const id = String(args.id ?? "").trim();
      if (!id) throw new Error("memory_delete: `id` is required");
      await client.delete(id);
      return { text: `Memory ${id} deleted.` };
    },
  })));

  return () => {
    for (const dispose of disposers.splice(0).reverse()) dispose();
  };
}
