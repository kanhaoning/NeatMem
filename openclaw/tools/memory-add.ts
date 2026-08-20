import { Type } from "@sinclair/typebox";
import type { AddOptions } from "../types.ts";
import { isSubagentSession } from "../isolation.ts";
import type { ToolDeps } from "./index.ts";

// NOTE: This tool is intentionally NOT registered (see tools/index.ts).
// It is kept for the "user explicitly asks to remember X" scenario and may
// return to the model-visible surface once the pipeline offers an immediate
// recall path (conversation search). Direct store (infer: false): the caller
// has already done the triage, so facts bypass extraction and are effective
// immediately.
export function createMemoryAddTool(deps: ToolDeps) {
  const { provider, resolveUserId, getCurrentSessionId } = deps;

  return {
    name: "memory_add",
    label: "Memory Add",
    description:
      "Store one fact verbatim, effective immediately. Use ONLY when: the user explicitly asks you to remember something; or an exceptionally important detail automatic capture might miss. Routine chat content is captured automatically — do not store it manually.",
    parameters: Type.Object({
      text: Type.Optional(Type.String({ description: "Single fact to remember" })),
      facts: Type.Optional(Type.Array(Type.String(), { description: "Array of facts to store" })),
      userId: Type.Optional(Type.String({ description: "User ID to scope this memory" })),
      agentId: Type.Optional(Type.String({ description: "Agent ID namespace" })),
      metadata: Type.Optional(Type.Record(Type.String(), Type.Unknown(), { description: "Additional metadata" })),
      longTerm: Type.Optional(Type.Boolean({ description: "Long-term (default: true). Set false for session-scoped." })),
    }),

    async execute(_toolCallId: string, params: Record<string, unknown>) {
      const p = params as {
        text?: string; facts?: string[];
        userId?: string; agentId?: string; metadata?: Record<string, unknown>; longTerm?: boolean;
      };

      const allFacts: string[] = p.facts?.length ? p.facts : (p.text ? [p.text] : []);
      if (allFacts.length === 0) {
        return { content: [{ type: "text", text: "No facts provided. Pass 'text' or 'facts' array." }], details: { error: "missing_facts" } };
      }

      try {
        const currentSessionId = getCurrentSessionId();

        if (isSubagentSession(currentSessionId)) {
          return { content: [{ type: "text", text: "Memory storage is not available in subagent sessions." }], details: { error: "subagent_blocked" } };
        }

        const uid = resolveUserId({ agentId: p.agentId, userId: p.userId });
        const runId = !(p.longTerm ?? true) && currentSessionId ? currentSessionId : undefined;

        const addOpts: AddOptions = {
          user_id: uid, source: "OPENCLAW", infer: false,
          deduced_memories: allFacts, metadata: p.metadata ?? {},
          output_format: "v1.1",
        };
        if (runId) addOpts.run_id = runId;

        const result = await provider.add([{ role: "user", content: allFacts.join("\n") }], addOpts);
        const count = result.results?.length ?? 0;

        return {
          content: [{ type: "text", text: `Stored ${allFacts.length} fact(s): ${allFacts.map(f => `"${f.slice(0, 60)}${f.length > 60 ? "..." : ""}"`).join(", ")}` }],
          details: { action: "stored", factCount: allFacts.length, stored: count, results: result.results },
        };
      } catch (err) {
        return { content: [{ type: "text", text: `Memory add failed: ${String(err)}` }], details: { error: String(err) } };
      }
    },
  };
}
