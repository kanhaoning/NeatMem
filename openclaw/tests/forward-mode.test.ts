/**
 * Tests for the queue-mode forward path in registerHooks (index.ts):
 * forward progress tracking, 404 permanent fallback, flush ordering.
 */
import { describe, it, expect, vi } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import {
  registerHooks,
  loadForwardProgress,
  saveForwardProgress,
} from "../index.ts";
import { NotFoundError } from "../backend/base.ts";

// ---------------------------------------------------------------------------
// Harness: capture registered hook handlers with a mock api
// ---------------------------------------------------------------------------

type HookHandler = (event: any, ctx: any) => unknown;

function setup(overrides?: {
  backend?: Partial<Record<"addMessages" | "flush", unknown>>;
  /** Reuse an existing HOME to simulate gateway plugin re-registration. */
  home?: string;
}) {
  // Fresh HOME per setup so the persisted forward-progress file never
  // leaks state between tests (resolved lazily at registerHooks time).
  process.env.HOME =
    overrides?.home ?? fs.mkdtempSync(path.join(os.tmpdir(), "fw-home-"));
  const handlers = new Map<string, HookHandler>();
  const api: any = {
    on: (name: string, fn: HookHandler) => handlers.set(name, fn),
    logger: { info: vi.fn(), warn: vi.fn() },
  };
  const provider: any = {
    add: vi.fn(async () => ({ results: [{ id: "m1" }] })),
  };
  const backend: any = {
    addMessages: vi.fn(async () => ({})),
    flush: vi.fn(async () => ({})),
    ...overrides?.backend,
  };
  const cfg: any = {
    autoCapture: true,
    autoRecall: false,
    userId: "test-user",
    topK: 5,
    searchThreshold: 0.3,
  };
  registerHooks(
    api,
    provider,
    backend,
    cfg,
    (sessionKey?: string) => `uid:${sessionKey ?? "none"}`,
    () => ({ user_id: "test-user" }) as any,
    (() => ({ user_id: "test-user" })) as any,
    { setCurrentSessionId: vi.fn() },
  );
  return { handlers, api, provider, backend };
}

const CTX = { sessionKey: "agent:main:main", trigger: "user" };

function userMsg(text: string) {
  return { role: "user", content: text };
}
function assistantMsg(text: string) {
  return { role: "assistant", content: text };
}
/** Assistant message carrying a memory-tool call plus optional text. */
function assistantToolMsg(tool: string, text = "done") {
  return {
    role: "assistant",
    content: [
      { type: "tool_use", name: tool, input: {} },
      { type: "text", text },
    ],
  };
}

async function settle() {
  // Let the forward promise chain drain.
  for (let i = 0; i < 10; i++) await Promise.resolve();
}

// ---------------------------------------------------------------------------
// Forward progress
// ---------------------------------------------------------------------------
describe("forward progress", () => {
  it("forwards only the new delta on each turn", async () => {
    const { handlers, backend } = setup();
    const agentEnd = handlers.get("agent_end")!;

    await agentEnd(
      { success: true, messages: [userMsg("u1"), assistantMsg("a1")] },
      CTX,
    );
    await settle();
    expect(backend.addMessages).toHaveBeenCalledTimes(1);
    expect(backend.addMessages.mock.calls[0][0]).toEqual([
      { role: "user", content: "u1" },
      { role: "assistant", content: "a1" },
    ]);

    await agentEnd(
      {
        success: true,
        messages: [userMsg("u1"), assistantMsg("a1"), userMsg("u2"), assistantMsg("a2")],
      },
      CTX,
    );
    await settle();
    expect(backend.addMessages).toHaveBeenCalledTimes(2);
    expect(backend.addMessages.mock.calls[1][0]).toEqual([
      { role: "user", content: "u2" },
      { role: "assistant", content: "a2" },
    ]);
  });

  it("resets progress when the snapshot shrinks (compaction / new session)", async () => {
    const { handlers, backend } = setup();
    const agentEnd = handlers.get("agent_end")!;

    await agentEnd(
      {
        success: true,
        messages: [userMsg("u1"), assistantMsg("a1"), userMsg("u2"), assistantMsg("a2")],
      },
      CTX,
    );
    await settle();
    expect(backend.addMessages).toHaveBeenCalledTimes(1);

    // Snapshot shrank from 4 to 2: forward the whole new snapshot.
    await agentEnd(
      { success: true, messages: [userMsg("u3"), assistantMsg("a3")] },
      CTX,
    );
    await settle();
    expect(backend.addMessages).toHaveBeenCalledTimes(2);
    expect(backend.addMessages.mock.calls[1][0]).toEqual([
      { role: "user", content: "u3" },
      { role: "assistant", content: "a3" },
    ]);
  });

  it("forwards user-side only on aborted turns", async () => {
    const { handlers, backend } = setup();
    const agentEnd = handlers.get("agent_end")!;

    await agentEnd(
      { success: true, messages: [userMsg("u1"), assistantMsg("a1")] },
      CTX,
    );
    await settle();

    // Aborted: snapshot gained only the interrupted turn's user message.
    await agentEnd(
      { success: false, messages: [userMsg("u1"), assistantMsg("a1"), userMsg("u2")] },
      CTX,
    );
    await settle();
    expect(backend.addMessages).toHaveBeenCalledTimes(2);
    expect(backend.addMessages.mock.calls[1][0]).toEqual([
      { role: "user", content: "u2" },
    ]);
  });

  it("skips the delta when the agent used memory-mutating tools this turn", async () => {
    const { handlers, backend } = setup();
    const agentEnd = handlers.get("agent_end")!;

    await agentEnd(
      { success: true, messages: [userMsg("u1"), assistantToolMsg("memory_update")] },
      CTX,
    );
    await settle();
    expect(backend.addMessages).not.toHaveBeenCalled();

    // Progress still advanced: next turn forwards only its own delta.
    await agentEnd(
      {
        success: true,
        messages: [userMsg("u1"), assistantToolMsg("memory_update"), userMsg("u2"), assistantMsg("a2")],
      },
      CTX,
    );
    await settle();
    expect(backend.addMessages).toHaveBeenCalledTimes(1);
    expect(backend.addMessages.mock.calls[0][0]).toEqual([
      { role: "user", content: "u2" },
      { role: "assistant", content: "a2" },
    ]);
  });
});

// ---------------------------------------------------------------------------
// 404 permanent fallback
// ---------------------------------------------------------------------------
describe("404 fallback", () => {
  it("falls back to legacy infer-add permanently after one 404", async () => {
    const { handlers, backend, provider } = setup({
      backend: {
        addMessages: vi.fn(async () => {
          throw new NotFoundError("/v1/messages/add/");
        }),
      },
    });
    const agentEnd = handlers.get("agent_end")!;

    const longUser =
      "u1 — this is a sufficiently long user message to pass the legacy fifty character floor";
    await agentEnd(
      { success: true, messages: [userMsg(longUser), assistantMsg("a1")] },
      CTX,
    );
    await settle();
    expect(backend.addMessages).toHaveBeenCalledTimes(1);
    // Legacy path kicked in for this turn.
    expect(provider.add).toHaveBeenCalledTimes(1);

    await agentEnd(
      {
        success: true,
        messages: [userMsg(longUser), assistantMsg("a1"), userMsg(longUser), assistantMsg("a2")],
      },
      CTX,
    );
    await settle();
    // No second addMessages attempt; straight to legacy.
    expect(backend.addMessages).toHaveBeenCalledTimes(1);
    expect(provider.add).toHaveBeenCalledTimes(2);
  });

  it("does not fall back on non-404 errors and retries next turn", async () => {
    let failOnce = true;
    const { handlers, backend, provider } = setup({
      backend: {
        addMessages: vi.fn(async () => {
          if (failOnce) {
            failOnce = false;
            throw new Error("HTTP 500: boom");
          }
          return {};
        }),
      },
    });
    const agentEnd = handlers.get("agent_end")!;

    await agentEnd(
      { success: true, messages: [userMsg("u1"), assistantMsg("a1")] },
      CTX,
    );
    await settle();
    expect(backend.addMessages).toHaveBeenCalledTimes(1);
    expect(provider.add).not.toHaveBeenCalled();

    // Progress rolled back: next turn re-forwards the failed delta plus its own.
    await agentEnd(
      {
        success: true,
        messages: [userMsg("u1"), assistantMsg("a1"), userMsg("u2"), assistantMsg("a2")],
      },
      CTX,
    );
    await settle();
    expect(backend.addMessages).toHaveBeenCalledTimes(2);
    expect(backend.addMessages.mock.calls[1][0]).toEqual([
      { role: "user", content: "u1" },
      { role: "assistant", content: "a1" },
      { role: "user", content: "u2" },
      { role: "assistant", content: "a2" },
    ]);
    expect(provider.add).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// Flush ordering
// ---------------------------------------------------------------------------
describe("flush", () => {
  it("session_end flush is queued after pending adds", async () => {
    const order: string[] = [];
    let releaseAdd!: () => void;
    const { handlers, backend } = setup({
      backend: {
        addMessages: vi.fn(
          () =>
            new Promise((resolve) => {
              releaseAdd = () => {
                order.push("add");
                resolve({});
              };
            }),
        ),
        flush: vi.fn(async () => {
          order.push("flush");
          return {};
        }),
      },
    });
    const agentEnd = handlers.get("agent_end")!;
    const sessionEnd = handlers.get("session_end")!;

    await agentEnd(
      { success: true, messages: [userMsg("u1"), assistantMsg("a1")] },
      CTX,
    );
    // Fire session_end while the add is still in flight.
    sessionEnd(
      { sessionKey: CTX.sessionKey, reason: "new" },
      CTX,
    );
    await settle();
    expect(backend.flush).toHaveBeenCalledTimes(0); // still blocked behind add
    releaseAdd();
    await settle();
    expect(backend.flush).toHaveBeenCalledTimes(1);
    expect(order).toEqual(["add", "flush"]);
  });

  it("flushes the previous scope when the sessionKey switches", async () => {
    const { handlers, backend } = setup();
    const beforePromptBuild = handlers.get("before_prompt_build")!;

    await beforePromptBuild({ prompt: "hello there" }, CTX);
    await settle();
    expect(backend.flush).not.toHaveBeenCalled();

    await beforePromptBuild(
      { prompt: "hello there" },
      { sessionKey: "agent:other:main", trigger: "user" },
    );
    await settle();
    expect(backend.flush).toHaveBeenCalledTimes(1);
    expect(backend.flush.mock.calls[0][0]).toEqual({
      userId: `uid:${CTX.sessionKey}`,
      runId: CTX.sessionKey,
    });
  });

  it("flushes persisted scopes from a previous plugin instance on first sighting", async () => {
    // First instance: forward one turn, progress persisted to disk.
    const first = setup();
    await first.handlers.get("agent_end")!(
      { success: true, messages: [userMsg("u1"), assistantMsg("a1")] },
      CTX,
    );
    await settle();
    expect(first.backend.addMessages).toHaveBeenCalledTimes(1);

    // Gateway re-registers the plugin (new TUI connect): fresh in-memory
    // state, same HOME. A turn on a different sessionKey must flush the
    // previous instance's scope.
    const second = setup({ home: process.env.HOME });
    await second.handlers.get("before_prompt_build")!(
      { prompt: "hello there" },
      { sessionKey: "agent:main:tui-new", trigger: "user" },
    );
    await settle();
    expect(second.backend.flush).toHaveBeenCalledTimes(1);
    expect(second.backend.flush.mock.calls[0][0]).toEqual({
      userId: `uid:${CTX.sessionKey}`,
      runId: CTX.sessionKey,
    });

    // And progress survives: a turn back on the old scope forwards only
    // the delta, not the whole history.
    await second.handlers.get("agent_end")!(
      {
        success: true,
        messages: [userMsg("u1"), assistantMsg("a1"), userMsg("u2"), assistantMsg("a2")],
      },
      CTX,
    );
    await settle();
    expect(second.backend.addMessages).toHaveBeenCalledTimes(1);
    expect(second.backend.addMessages.mock.calls[0][0]).toEqual([
      { role: "user", content: "u2" },
      { role: "assistant", content: "a2" },
    ]);
  });

  it("logs but does not crash when flush 404s", async () => {
    const { handlers, api } = setup({
      backend: {
        flush: vi.fn(async () => {
          throw new NotFoundError("/v1/messages/flush/");
        }),
      },
    });
    const sessionEnd = handlers.get("session_end")!;
    sessionEnd({ sessionKey: CTX.sessionKey, reason: "new" }, CTX);
    await settle();
    expect(api.logger.info).toHaveBeenCalledWith(
      expect.stringContaining("flush skipped"),
    );
  });
});

// ---------------------------------------------------------------------------
// Progress persistence (gateway re-registers plugins on TUI connect)
// ---------------------------------------------------------------------------
describe("forward progress persistence", () => {
  it("round-trips through the state file", () => {
    const file = path.join(
      fs.mkdtempSync(path.join(os.tmpdir(), "fw-prog-")),
      "state.json",
    );
    const map = new Map([
      ["agent:main:main", 12],
      ["agent:main:other", 3],
    ]);
    saveForwardProgress(map, file);
    expect(loadForwardProgress(file)).toEqual(map);
  });

  it("returns an empty map for a missing or corrupt file", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "fw-prog-"));
    const missing = path.join(dir, "nope.json");
    expect(loadForwardProgress(missing).size).toBe(0);
    const corrupt = path.join(dir, "corrupt.json");
    fs.writeFileSync(corrupt, "{not json");
    expect(loadForwardProgress(corrupt).size).toBe(0);
  });

  it("drops non-numeric values when loading", () => {
    const file = path.join(
      fs.mkdtempSync(path.join(os.tmpdir(), "fw-prog-")),
      "state.json",
    );
    fs.writeFileSync(
      file,
      JSON.stringify({ good: 5, bad: "x", negative: -1 }),
    );
    const loaded = loadForwardProgress(file);
    expect(loaded.get("good")).toBe(5);
    expect(loaded.has("bad")).toBe(false);
    expect(loaded.has("negative")).toBe(false);
  });
});
