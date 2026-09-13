/**
 * Automatic recall on `agent/pre-step`: one bounded search per direct-user
 * turn, injected as a source-labelled user message so the context is
 * reconstructable from the session log (model-visible ⟺ logged).
 *
 * Fail-open everywhere: timeout, server down, or malformed responses return
 * the downstream waterfall decision unmodified.
 */

import type { PreStepDecision } from "@deepseek-ai/dsh-agent";
import { createUserMessage, type UserMessage } from "@deepseek-ai/dsh-llm";

import type { MemoryHit, NeatMemClient } from "./client.js";
import { waitForDeadline } from "./deadline.js";

export const RECALL_SOURCE_PLUGIN = "neatmem-dsh";
/** Product-level SLA for foreground recall, regardless of configuration. */
export const MAX_FOREGROUND_RECALL_MS = 3_000;

export interface RecallOptions {
  client: NeatMemClient;
  userId: string;
  agentId: string;
  topK: number;
  searchThreshold: number;
  recallTimeoutMs: number;
  contextMaxChars: number;
  onInfo: (message: string) => void;
  onWarn: (message: string, error?: unknown) => void;
}

/** Extract joined text from text-type content blocks. */
function textFromContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter(
      (block): block is { type: "text"; text: string } =>
        typeof block === "object" && block !== null &&
        (block as { type?: unknown }).type === "text" &&
        typeof (block as { text?: unknown }).text === "string",
    )
    .map((block) => block.text)
    .join("\n");
}

/** Only direct human input (source `user`) triggers recall — never plugin or tool messages. */
function userTextFrom(messages: readonly UserMessage[]): string {
  return messages
    .filter((message) => message.source.kind === "user")
    .map((message) => textFromContent(message.content))
    .filter(Boolean)
    .join("\n")
    .trim();
}

/** Client-side score filtering, mirroring the openclaw plugin's recall path. */
function filterHits(hits: MemoryHit[], threshold: number, topK: number): MemoryHit[] {
  const floor = Math.max(threshold, 0.1);
  let kept = hits.filter((hit) => (hit.score ?? 0) >= floor);
  if (kept.length > 1) {
    const topScore = kept[0]?.score ?? 0;
    if (topScore > 0) kept = kept.filter((hit) => (hit.score ?? 0) >= topScore * 0.5);
  }
  return kept.slice(0, topK);
}

export function renderRecallContext(
  hits: MemoryHit[],
  userId: string,
  maxChars: number,
): string {
  if (hits.length === 0) return "";
  const lines = hits.map(
    (hit) =>
      `- ${String(hit.memory ?? "").trim()}` +
      (hit.categories?.length ? ` [${hit.categories.join(", ")}]` : ""),
  );
  const body = [
    `<neatmem_context>`,
    `The following are stored memories for user "${userId}". Use them to personalize your response.`,
    ...lines,
    `</neatmem_context>`,
  ].join("\n");
  return body.length <= maxChars ? body : `${body.slice(0, maxChars - 1)}…`;
}

export function createRecallHandler(options: RecallOptions) {
  // One recall per session+turn; pre-step can fire repeatedly for a turn.
  const attempted = new Set<string>();

  return async function onPreStep(
    payload: {
      agent: { id: string; session: { id: string } };
      messages: UserMessage[];
      turn: number;
      step: number;
      signal: AbortSignal;
    },
    next: () => Promise<PreStepDecision>,
  ): Promise<PreStepDecision> {
    // `next()` is the authoritative waterfall decision; recall only appends.
    const decision = await next();
    if (decision.kind !== "enter" || payload.step !== 1) return decision;

    const userText = userTextFrom(decision.messages);
    if (!userText) return decision;

    const key = `${payload.agent.session.id}:${payload.turn}`;
    if (attempted.has(key)) return decision;
    attempted.add(key);

    const budgetMs = Math.min(options.recallTimeoutMs, MAX_FOREGROUND_RECALL_MS);
    try {
      if (payload.signal.aborted) return decision;
      const deadlineAt = Date.now() + budgetMs;
      const hits = await waitForDeadline(
        options.client.search(userText, {
          userId: options.userId,
          agentId: options.agentId || undefined,
          // Wider candidate pool; client-side filtering caps at topK.
          topK: Math.max(options.topK * 2, 10),
          threshold: options.searchThreshold,
          // Auto-recall takes the fast path; rerank stays a tool-level choice.
          rerank: false,
        }),
        {
          deadlineAt,
          signal: payload.signal,
          timeoutMessage: `neatmem recall exceeded ${budgetMs}ms`,
        },
      );
      const kept = filterHits(hits, options.searchThreshold, options.topK);
      const context = renderRecallContext(kept, options.userId, options.contextMaxChars);
      options.onInfo(
        `recall session=${payload.agent.session.id} turn=${payload.turn} hits=${kept.length} chars=${context.length}`,
      );
      if (!context || payload.signal.aborted) return decision;
      return {
        kind: "enter",
        // Accepted user input first, then source-labelled context — same
        // ordering as DSH's native context injection.
        messages: [
          ...decision.messages,
          createUserMessage({
            content: [{ type: "text", text: context }],
            source: { kind: "plugin", plugin: RECALL_SOURCE_PLUGIN, form: "recall" },
          }),
        ],
      };
    } catch (error) {
      // Memory is optional: a retrieval failure must not block the step.
      options.onWarn(
        `recall failed for session ${payload.agent.session.id}, turn ${payload.turn}`,
        error,
      );
      return decision;
    }
  };
}
