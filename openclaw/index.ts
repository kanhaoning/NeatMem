/**
 * OpenClaw Memory (NeatMem) Plugin
 *
 * Long-term memory via a local NeatMem server — talks to a self-hosted
 * NeatMem backend over the mem0-compatible REST API.
 *
 * Features:
 * - 5 core tools: memory_search, memory_get, memory_list,
 *   memory_update, memory_delete
 * - Short-term (session-scoped) and long-term (user-scoped) memory
 * - Auto-recall: injects relevant memories (both scopes) before each agent turn
 * - Auto-capture: stores key facts scoped to the current session after each agent turn
 * - Per-agent isolation: multi-agent setups write/read from separate userId namespaces
 *   automatically via sessionKey routing (zero breaking changes for single-agent setups)
 * - CLI: openclaw neatmem search, openclaw neatmem status
 * - Writes are pipeline-driven (auto-capture); the model has a read-mostly
 *   tool surface. memory_add stays implemented but is not registered
 *   (2026-08-20, see docs/internal-notes/20260820-openclaw-plugin-slim-down-plan.md)
 */

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import type {
  Mem0Config,
  Mem0Provider,
  AddOptions,
  SearchOptions,
} from "./types.ts";
import { createProvider } from "./providers.ts";
import { mem0ConfigSchema } from "./config.ts";
import type { FileConfig } from "./config.ts";
import { filterMessagesForExtraction } from "./filtering.ts";
import {
  effectiveUserId,
  agentUserId,
  resolveUserId,
  isNonInteractiveTrigger,
  isSubagentSession,
} from "./isolation.ts";
import { sanitizeQuery } from "./recall.ts";
import { PlatformBackend } from "./backend/platform.ts";
import { NotFoundError, type Backend } from "./backend/base.ts";
import { registerCliCommands } from "./cli/commands.ts";
import { readPluginAuth } from "./cli/config-file.ts";
import { registerAllTools } from "./tools/index.ts";
import type { ToolDeps } from "./tools/index.ts";

// ============================================================================
// Re-exports (for tests and external consumers)
// ============================================================================

export {
  extractAgentId,
  effectiveUserId,
  agentUserId,
  resolveUserId,
  isNonInteractiveTrigger,
  isSubagentSession,
} from "./isolation.ts";
export {
  isNoiseMessage,
  isGenericAssistantMessage,
  stripNoiseFromContent,
  filterMessagesForExtraction,
} from "./filtering.ts";
export { mem0ConfigSchema } from "./config.ts";
export type { FileConfig } from "./config.ts";
export { createProvider } from "./providers.ts";

// ============================================================================
// Helpers
// ============================================================================

// Computed lazily so tests can relocate HOME before registerHooks runs.
function defaultProgressFile(): string {
  return path.join(
    os.homedir(),
    ".openclaw",
    "memory",
    "neatmem-forward-progress.json",
  );
}

// Exported for tests.
export function loadForwardProgress(
  filePath: string = defaultProgressFile(),
): Map<string, number> {
  try {
    const raw = fs.readFileSync(filePath, "utf-8");
    const obj = JSON.parse(raw) as Record<string, unknown>;
    const map = new Map<string, number>();
    for (const [k, v] of Object.entries(obj)) {
      if (typeof v === "number" && Number.isFinite(v) && v >= 0) {
        map.set(k, v);
      }
    }
    return map;
  } catch {
    // Missing or corrupt file: start empty.
    return new Map<string, number>();
  }
}

// Exported for tests.
export function saveForwardProgress(
  progress: Map<string, number>,
  filePath: string = defaultProgressFile(),
): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  const tmp = `${filePath}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(Object.fromEntries(progress)));
  fs.renameSync(tmp, filePath);
}

// ============================================================================
// Plugin Definition
// ============================================================================

const memoryPlugin = definePluginEntry({
  id: "openclaw-neatmem",
  name: "Memory (NeatMem)",
  description: "NeatMem memory backend — local, controllable, mem0-compatible",

  register(api: OpenClawPluginApi) {
    // Read auth from openclaw.json plugin config (picks up post-startup login).
    // This is the single source of truth — set via `openclaw neatmem init`.
    const pluginAuth = readPluginAuth();
    const fileConfig: FileConfig = {
      apiKey: pluginAuth.apiKey,
      baseUrl: pluginAuth.baseUrl,
    };
    const cfg = mem0ConfigSchema.parse(api.pluginConfig, fileConfig);

    if (cfg.needsSetup) {
      api.logger.warn(
        "openclaw-neatmem: API key not configured. Memory features are disabled.\n" +
          "  To set up, run:\n" +
          "  openclaw neatmem init",
      );

      // Register CLI even without API key — init command must be available
      // to bootstrap configuration. Pass nulls for backend/provider since
      // only the init subcommand works without auth.
      registerCliCommands(
        api,
        null as any,
        null as any,
        cfg,
        () => cfg.userId,
        (id: string) => `${cfg.userId}:agent:${id}`,
        () => ({ user_id: cfg.userId, top_k: cfg.topK }),
        () => undefined,
      );

      api.registerService({
        id: "openclaw-neatmem",
        start: () => {
          api.logger.info("openclaw-neatmem: waiting for API key configuration");
        },
        stop: () => {},
      });
      return;
    }

    const provider = createProvider(cfg);

    const backend: Backend = new PlatformBackend({
      apiKey: cfg.apiKey!,
      baseUrl: cfg.baseUrl ?? "http://localhost:8790",
    });

    // Shared mutable state — declared together before any closures capture them.
    let currentSessionId: string | undefined;

    // ========================================================================
    // Per-agent isolation helpers (thin wrappers around exported functions)
    // ========================================================================
    const _effectiveUserId = (sessionKey?: string) =>
      effectiveUserId(cfg.userId, sessionKey);
    const _agentUserId = (id: string) => agentUserId(cfg.userId, id);
    const _resolveUserId = (opts: { agentId?: string; userId?: string }) =>
      resolveUserId(cfg.userId, opts, currentSessionId);

    api.logger.info(
      `openclaw-neatmem: registered (user: ${cfg.userId}, autoRecall: ${cfg.autoRecall}, autoCapture: ${cfg.autoCapture})`,
    );

    // Helper: build add options
    function buildAddOptions(
      userIdOverride?: string,
      runId?: string,
      sessionKey?: string,
    ): AddOptions {
      const opts: AddOptions = {
        user_id: userIdOverride || _effectiveUserId(sessionKey),
        source: "OPENCLAW",
        output_format: "v1.1",
      };
      if (runId) opts.run_id = runId;
      return opts;
    }

    // Helper: build search options (recall config overrides defaults)
    function buildSearchOptions(
      userIdOverride?: string,
      limit?: number,
      runId?: string,
      sessionKey?: string,
    ): SearchOptions {
      const recallCfg = cfg.recall;
      const opts: SearchOptions = {
        user_id: userIdOverride || _effectiveUserId(sessionKey),
        top_k: limit ?? cfg.topK,
        limit: limit ?? cfg.topK,
        threshold: recallCfg?.threshold ?? cfg.searchThreshold,
        keyword_search: recallCfg?.keywordSearch !== false,
        reranking: recallCfg?.rerank !== false,
        source: "OPENCLAW",
      };
      if (recallCfg?.filterMemories) opts.filter_memories = true;
      if (runId) opts.run_id = runId;
      return opts;
    }

    // ========================================================================
    // Tools (modular — each tool in its own file under tools/)
    // ========================================================================

    const toolDeps: ToolDeps = {
      api,
      provider,
      cfg,
      backend,
      resolveUserId: _resolveUserId,
      effectiveUserId: _effectiveUserId,
      agentUserId: _agentUserId,
      buildAddOptions,
      buildSearchOptions,
      getCurrentSessionId: () => currentSessionId,
    };
    registerAllTools(toolDeps);

    // ========================================================================
    // CLI Commands
    // ========================================================================

    registerCliCommands(
      api,
      backend,
      provider,
      cfg,
      _effectiveUserId,
      _agentUserId,
      buildSearchOptions,
      () => currentSessionId,
    );

    // ========================================================================
    // Lifecycle Hooks
    // ========================================================================

    registerHooks(
      api,
      provider,
      backend,
      cfg,
      _effectiveUserId,
      buildAddOptions,
      buildSearchOptions,
      {
        setCurrentSessionId: (id: string) => {
          currentSessionId = id;
        },
      },
    );

    // ========================================================================
    // Service
    // ========================================================================

    api.registerService({
      id: "openclaw-neatmem",
      start: () => {
        api.logger.info(
          `openclaw-neatmem: initialized (user: ${cfg.userId}, autoRecall: ${cfg.autoRecall}, autoCapture: ${cfg.autoCapture})`,
        );
      },
      stop: () => {
        api.logger.info("openclaw-neatmem: stopped");
      },
    });
  },
});

// ============================================================================
// Lifecycle Hook Registration
// ============================================================================

// Exported for tests (see tests/forward-mode.test.ts).
export function registerHooks(
  api: OpenClawPluginApi,
  provider: Mem0Provider,
  backend: Backend,
  cfg: Mem0Config,
  _effectiveUserId: (sessionKey?: string) => string,
  buildAddOptions: (
    userIdOverride?: string,
    runId?: string,
    sessionKey?: string,
  ) => AddOptions,
  buildSearchOptions: (
    userIdOverride?: string,
    limit?: number,
    runId?: string,
    sessionKey?: string,
  ) => SearchOptions,
  session: {
    setCurrentSessionId: (id: string) => void;
  },
) {
  // ========================================================================
  // Auto-recall + auto-capture (single pipeline mode; skills mode removed in 2.0.0)
  // ========================================================================

  // Track last seen session ID to detect actual new sessions (not every turn)
  let lastRecallSessionId: string | undefined;

  // Auto-recall: inject relevant memories before prompt is built
  if (cfg.autoRecall) {
    const RECALL_TIMEOUT_MS = 8_000;

    api.on("before_prompt_build", async (event: any, ctx: any) => {
      if (!event.prompt || event.prompt.length < 5) return;

      // Skip non-interactive triggers (cron, heartbeat, automation)
      const trigger = (ctx as any)?.trigger ?? undefined;
      const sessionId = (ctx as any)?.sessionKey ?? undefined;
      if (isNonInteractiveTrigger(trigger, sessionId)) {
        api.logger.info(
          "openclaw-neatmem: skipping recall for non-interactive trigger",
        );
        return;
      }

      const promptLower = event.prompt.toLowerCase();
      const isSystemPrompt =
        promptLower.includes("a new session was started") ||
        promptLower.includes("session startup sequence") ||
        promptLower.includes("/new or /reset") ||
        promptLower.startsWith("run your session");
      if (isSystemPrompt) {
        api.logger.info(
          "openclaw-neatmem: skipping recall for system/bootstrap prompt",
        );
        return;
      }

      // Update shared state for tools (best-effort — tools don't have ctx)
      if (sessionId) session.setCurrentSessionId(sessionId);

      // Detect actual new session (first turn with a different sessionKey)
      const isNewSession =
        sessionId !== undefined && sessionId !== lastRecallSessionId;
      if (sessionId) lastRecallSessionId = sessionId;

      // Subagents have ephemeral UUIDs — their namespace is always empty.
      // Search the parent (main) user namespace instead so subagents get
      // the user's long-term context.
      const isSubagent = isSubagentSession(sessionId);
      const recallSessionKey = isSubagent ? undefined : sessionId;

      // Strip OpenClaw envelope prefix and sender metadata before searching
      const cleanPrompt = sanitizeQuery(event.prompt);

      const recallStart = Date.now();
      const recallWork = async () => {
        // Single search with a reasonable candidate pool
        const recallTopK = Math.max((cfg.topK ?? 5) * 2, 10);

        // Search long-term memories (user-scoped; subagents read from parent namespace)
        let longTermResults = await provider.search(
          cleanPrompt,
          buildSearchOptions(
            undefined,
            recallTopK,
            undefined,
            recallSessionKey,
          ),
        );

        // Client-side threshold filter for auto-recall.
        // Minimum 0.1: trust the backend's threshold filtering rather than
        // imposing a high client-side floor. The original 0.6 floor assumed
        // raw vector similarity scores, but with LLM reranker producing
        // definitive 0/1 scores or backend-side threshold enforcement,
        // a high floor causes false negatives.
        const recallThreshold = Math.max(cfg.searchThreshold, 0.1);
        longTermResults = longTermResults.filter(
          (r) => (r.score ?? 0) >= recallThreshold,
        );

        // Dynamic thresholding: drop memories scoring less than 50% of
        // the top result's score to filter out the long tail of weak matches
        if (longTermResults.length > 1) {
          const topScore = longTermResults[0]?.score ?? 0;
          if (topScore > 0) {
            longTermResults = longTermResults.filter(
              (r) => (r.score ?? 0) >= topScore * 0.5,
            );
          }
        }

        // Only broaden for genuinely new sessions with short prompts
        // (cold-start blindness). Skip on subsequent turns to save API calls.
        if (isNewSession && cleanPrompt.length < 100) {
          const broadOpts = buildSearchOptions(
            undefined,
            5,
            undefined,
            recallSessionKey,
          );
          broadOpts.threshold = 0.1;
          const broadResults = await provider.search(
            "recent decisions, preferences, active projects, and configuration",
            broadOpts,
          );
          const existingIds = new Set(longTermResults.map((r) => r.id));
          for (const r of broadResults) {
            if (!existingIds.has(r.id)) {
              longTermResults.push(r);
            }
          }
        }

        // Cap at configured topK after filtering
        longTermResults = longTermResults.slice(0, cfg.topK);

        if (longTermResults.length === 0) return undefined;

        // Build context with clear labels
        const memoryContext = longTermResults
          .map(
            (r) =>
              `- ${r.memory}${r.categories?.length ? ` [${r.categories.join(", ")}]` : ""}`,
          )
          .join("\n");

        api.logger.info(
          `openclaw-neatmem: injecting ${longTermResults.length} memories into context`,
        );

        const preamble = isSubagent
          ? `The following are stored memories for user "${cfg.userId}". You are a subagent — use these memories for context but do not assume you are this user.`
          : `The following are stored memories for user "${cfg.userId}". Use them to personalize your response:`;

        return {
          prependContext: `<relevant-memories>\n${preamble}\n${memoryContext}\n</relevant-memories>`,
        };
      };

      try {
        const timeout = new Promise<undefined>((resolve) => {
          setTimeout(() => resolve(undefined), RECALL_TIMEOUT_MS);
        });
        const result = await Promise.race([
          recallWork(),
          timeout.then(() => {
            api.logger.warn(
              `openclaw-neatmem: recall timed out after ${RECALL_TIMEOUT_MS}ms, skipping`,
            );
            return undefined;
          }),
        ]);
        return result;
      } catch (err) {
        api.logger.warn(`openclaw-neatmem: recall failed: ${String(err)}`);
      }
    });
  }

  // Auto-capture: forward raw conversation messages to the server
  // (queue mode). The server batches extraction (batch size / deadline);
  // the plugin only stores [user, assistant] text via /v1/messages/add/.
  // On servers predating the queue endpoints (404) the plugin permanently
  // falls back to the legacy per-turn infer=true extraction.
  if (cfg.autoCapture) {
    let forwardUnsupported = false;
    // Per-scope forward progress: agent_end delivers a full-session
    // snapshot, so the plugin must remember how far it has forwarded.
    // Persisted to disk: the gateway re-registers plugins when a new TUI
    // client connects (observed 2026-08-21), which would otherwise reset
    // progress and re-forward the entire history.
    const forwardProgress = loadForwardProgress();
    let progressSaveWarned = false;
    const persistForwardProgress = () => {
      try {
        saveForwardProgress(forwardProgress);
      } catch (err) {
        if (!progressSaveWarned) {
          progressSaveWarned = true;
          api.logger.warn(
            `openclaw-neatmem: failed to persist forward progress: ${String(err)}`,
          );
        }
      }
    };
    // Serialize forwards so the server receives messages in order.
    let forwardChain: Promise<void> = Promise.resolve();
    const enqueueForward = (work: () => Promise<void>) => {
      forwardChain = forwardChain.then(work);
    };

    const MEMORY_MUTATE_TOOLS = new Set([
      "memory_add",
      "memory_update",
      "memory_delete",
    ]);

    // Parse a snapshot into plain [user, assistant] text, stripping the
    // plugin's own <relevant-memories> injection and OpenClaw's Sender
    // metadata envelope. index = position in the original snapshot.
    const parseForwardMessages = (
      messages: unknown[],
    ): Array<{ role: string; content: string; index: number }> => {
      const parsed: Array<{ role: string; content: string; index: number }> =
        [];
      for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        if (!msg || typeof msg !== "object") continue;
        const msgObj = msg as Record<string, unknown>;

        const role = msgObj.role;
        if (role !== "user" && role !== "assistant") continue;

        let textContent = "";
        const content = msgObj.content;
        if (typeof content === "string") {
          textContent = content;
        } else if (Array.isArray(content)) {
          for (const block of content) {
            if (
              block &&
              typeof block === "object" &&
              "text" in block &&
              typeof (block as Record<string, unknown>).text === "string"
            ) {
              textContent +=
                (textContent ? "\n" : "") +
                ((block as Record<string, unknown>).text as string);
            }
          }
        }

        if (!textContent) continue;
        // Strip injected memory context, keep the actual user text
        if (textContent.includes("<relevant-memories>")) {
          textContent = textContent
            .replace(/<relevant-memories>[\s\S]*?<\/relevant-memories>\s*/g, "")
            .trim();
          if (!textContent) continue;
        }
        // Strip OpenClaw sender metadata prefix (prevents storing TUI identity as memory)
        if (
          textContent.includes("Sender") &&
          textContent.includes("untrusted metadata")
        ) {
          textContent = textContent
            .replace(
              /Sender\s*\(untrusted metadata\):\s*```json[\s\S]*?```\s*/gi,
              "",
            )
            .trim();
          if (!textContent) continue;
        }

        parsed.push({ role: role as string, content: textContent, index: i });
      }
      return parsed;
    };

    const snapshotUsedMemoryTool = (messages: unknown[], fromIndex: number) =>
      messages.slice(fromIndex).some((msg: any) => {
        if (msg?.role !== "assistant" || !Array.isArray(msg?.content))
          return false;
        return msg.content.some(
          (block: any) =>
            (block?.type === "tool_use" || block?.type === "toolCall") &&
            MEMORY_MUTATE_TOOLS.has(block.name),
        );
      });

    // Legacy per-turn extraction path (pre-queue-mode servers).
    const legacyCapture = (messages: unknown[], sessionId?: string): void => {
      if (snapshotUsedMemoryTool(messages, 0)) {
        api.logger.info(
          "openclaw-neatmem: skipping auto-capture — agent already used memory tools this turn",
        );
        return;
      }

      // Patterns indicating an assistant message contains a summary of
      // completed work — these are high-value for extraction and should
      // be included even if they fall outside the recent-message window.
      const SUMMARY_PATTERNS = [
        /## What I (Accomplished|Built|Updated)/i,
        /✅\s*(Done|Complete|All done)/i,
        /Here's (what I updated|the recap|a summary)/i,
        /### Changes Made/i,
        /Implementation Status/i,
        /All locked in\. Quick summary/i,
      ];

      const allParsed = parseForwardMessages(messages).map((m) => ({
        ...m,
        isSummary:
          m.role === "assistant" &&
          SUMMARY_PATTERNS.some((p) => p.test(m.content)),
      }));

      if (allParsed.length === 0) return;

      // Select messages: last 20 + any earlier summary messages,
      // sorted by original index to preserve chronological order.
      const recentWindow = 20;
      const recentCutoff = allParsed.length - recentWindow;

      const candidates: typeof allParsed = [];
      for (const msg of allParsed) {
        if (msg.isSummary && msg.index < recentCutoff) candidates.push(msg);
      }
      const seenIndices = new Set(candidates.map((m) => m.index));
      for (const msg of allParsed) {
        if (msg.index >= recentCutoff && !seenIndices.has(msg.index)) {
          candidates.push(msg);
        }
      }
      candidates.sort((a, b) => a.index - b.index);

      const selected = candidates.map((m) => ({
        role: m.role,
        content: m.content,
      }));

      // Apply noise filtering pipeline: drop noise, strip fragments, truncate
      const formattedMessages = filterMessagesForExtraction(selected);

      if (formattedMessages.length === 0) return;
      if (!formattedMessages.some((m) => m.role === "user")) return;
      const userContent = formattedMessages
        .filter((m) => m.role === "user")
        .map((m) => m.content)
        .join(" ");
      if (userContent.length < 50) {
        api.logger.info(
          "openclaw-neatmem: skipping capture — user content too short for meaningful extraction",
        );
        return;
      }

      // Inject a timestamp preamble so the extraction model can anchor
      // time-sensitive facts to a concrete date and attribute to the correct user
      const timestamp = new Date().toISOString().split("T")[0];
      formattedMessages.unshift({
        role: "system",
        content: `Current date: ${timestamp}. The user is identified as "${cfg.userId}". Extract durable facts from this conversation. Include this date when storing time-sensitive information.`,
      });

      const addOpts = buildAddOptions(undefined, sessionId, sessionId);
      provider
        .add(formattedMessages, addOpts)
        .then((result) => {
          const capturedCount = result.results?.length ?? 0;
          if (capturedCount > 0) {
            api.logger.info(
              `openclaw-neatmem: auto-captured ${capturedCount} memories (legacy path)`,
            );
          }
        })
        .catch((err) => {
          api.logger.warn(`openclaw-neatmem: capture failed: ${String(err)}`);
        });
    };

    api.on("agent_end", async (event, ctx) => {
      if (!event.messages || event.messages.length === 0) {
        return;
      }

      // Skip non-interactive triggers (cron, heartbeat, automation)
      const trigger = (ctx as any)?.trigger ?? undefined;
      const sessionId = (ctx as any)?.sessionKey ?? undefined;
      if (isNonInteractiveTrigger(trigger, sessionId)) {
        api.logger.info(
          "openclaw-neatmem: skipping capture for non-interactive trigger",
        );
        return;
      }

      // Skip capture for subagents — their ephemeral UUIDs create orphaned
      // namespaces that are never read again. The main agent's agent_end
      // hook captures the consolidated result including subagent output.
      if (isSubagentSession(sessionId)) {
        api.logger.info(
          "openclaw-neatmem: skipping capture for subagent (main agent captures consolidated result)",
        );
        return;
      }

      // Update shared state for tools (best-effort — tools don't have ctx)
      if (sessionId) session.setCurrentSessionId(sessionId);

      if (forwardUnsupported) {
        // Legacy path captures only successful turns.
        if (event.success) legacyCapture(event.messages, sessionId);
        return;
      }

      const parsed = parseForwardMessages(event.messages);
      if (parsed.length === 0) return;

      const scope = sessionId ?? "default";
      const prev = forwardProgress.get(scope) ?? 0;
      // Snapshot shrank (compaction, or a new session reusing the key) —
      // restart progress from zero.
      const start = parsed.length < prev ? 0 : prev;
      let delta = parsed.slice(start);
      // Aborted turn: the snapshot carries only the user side of the
      // interrupted turn (verified 2026-08-21, plan doc §3.0). Forward
      // user messages only — never a partial assistant reply.
      if (!event.success) {
        delta = delta.filter((m) => m.role === "user");
      }
      forwardProgress.set(scope, parsed.length);
      persistForwardProgress();
      if (delta.length === 0) return;

      // Skip the delta if the agent already wrote memory itself this turn
      // (update/delete are model-visible in some tool profiles).
      if (snapshotUsedMemoryTool(event.messages, delta[0]!.index)) {
        api.logger.info(
          "openclaw-neatmem: skipping forward — agent already used memory tools this turn",
        );
        return;
      }

      const userId = _effectiveUserId(sessionId);
      const forwardedCount = delta.length;
      enqueueForward(async () => {
        try {
          await backend.addMessages(
            delta.map((m) => ({ role: m.role, content: m.content })),
            { userId, runId: sessionId },
          );
          api.logger.info(
            `openclaw-neatmem: forwarded ${forwardedCount} messages (scope: ${scope})`,
          );
        } catch (err) {
          if (err instanceof NotFoundError) {
            forwardUnsupported = true;
            api.logger.warn(
              "openclaw-neatmem: /v1/messages/add/ not found (old server) — falling back to per-turn extraction",
            );
            if (event.success) legacyCapture(event.messages, sessionId);
          } else {
            // Roll progress back so the next turn re-forwards this delta.
            forwardProgress.set(scope, start);
            persistForwardProgress();
            api.logger.warn(
              `openclaw-neatmem: forward failed (retry next turn): ${String(err)}`,
            );
          }
        }
      });
    });

    // Flush queued messages at session boundaries so a conversation's tail
    // (below the server batch threshold) gets extracted before the next
    // session needs to recall it.
    const flushScope = (sessionKey: string, why: string) => {
      const userId = _effectiveUserId(sessionKey);
      enqueueForward(async () => {
        try {
          await backend.flush({ userId, runId: sessionKey });
          api.logger.info(
            `openclaw-neatmem: flushed scope ${sessionKey} (${why})`,
          );
        } catch (err) {
          // 404 = old server (the add path detects this permanently on its
          // own); log-only either way.
          api.logger.info(
            `openclaw-neatmem: flush skipped (${why}): ${String(err)}`,
          );
        }
      });
    };

    api.on("session_end", (event: any, ctx: any) => {
      if (forwardUnsupported) return;
      const sessionKey =
        event?.sessionKey ?? (ctx as any)?.sessionKey ?? undefined;
      if (
        !sessionKey ||
        isSubagentSession(sessionKey) ||
        isNonInteractiveTrigger(undefined, sessionKey)
      ) {
        return;
      }
      flushScope(sessionKey, `session_end: ${event?.reason ?? "unknown"}`);
    });

    // Fallback flush trigger: a sessionKey switch (channel/agent routing
    // change) may not emit session_end for the previous scope — TUI /new
    // creates a fresh sessionKey without one (observed 2026-08-21). Also,
    // on this process's first sighting of any sessionKey, flush the other
    // persisted scopes: the gateway re-registers plugins on TUI connect,
    // so scopes from the previous plugin instance would otherwise wait for
    // the server-side deadline. Flushing an active scope is harmless — it
    // only extracts the pending tail early.
    let lastCaptureSessionKey: string | undefined;
    api.on("before_prompt_build", async (_event: any, ctx: any) => {
      const sk = (ctx as any)?.sessionKey ?? undefined;
      if (!sk || sk === lastCaptureSessionKey) return;
      const prevKey = lastCaptureSessionKey;
      lastCaptureSessionKey = sk;
      if (forwardUnsupported) return;
      if (prevKey) {
        if (!isSubagentSession(prevKey) && !isNonInteractiveTrigger(undefined, prevKey)) {
          flushScope(prevKey, "sessionKey switch");
        }
        return;
      }
      for (const scope of forwardProgress.keys()) {
        if (
          scope !== sk &&
          !isSubagentSession(scope) &&
          !isNonInteractiveTrigger(undefined, scope)
        ) {
          flushScope(scope, "plugin re-registered");
        }
      }
    });
  }
}

export default memoryPlugin;
