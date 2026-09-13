/**
 * Turn capture on the `session/event` firehose. The hook is synchronous: it
 * only records events and enqueues writes. Each session's writes run on a
 * serial promise chain; every job is one cheap store-first POST to
 * `/v1/messages/add/` (extraction is the server's scheduler). Session
 * disposal enqueues a best-effort bounded flush; Cordis disposal drains the
 * queues within a bounded window.
 */

import type { Session, SessionEvent } from "@deepseek-ai/dsh-session";

import type { NeatMemClient } from "./client.js";

export interface CaptureOptions {
  client: NeatMemClient;
  userId: string;
  agentId: string;
  captureTimeoutMs: number;
  flushTimeoutMs: number;
  onInfo: (message: string) => void;
  onWarn: (message: string, error?: unknown) => void;
}

interface TurnCapture {
  userText: string[];
  assistantText: string[];
  queued: boolean;
}

interface SessionCapture {
  activeTurn?: number;
  turns: Map<number, TurnCapture>;
  /** A capture job was enqueued; worth a flush on close/dispose. */
  dirty: boolean;
  flushed: boolean;
}

function textBlocks(content: unknown): string {
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

export class CaptureQueue {
  private readonly options: CaptureOptions;
  private readonly sessions = new Map<string, SessionCapture>();
  private readonly pendingBySession = new Map<string, Promise<void>>();

  constructor(options: CaptureOptions) {
    this.options = options;
  }

  /** Synchronous firehose entry point — record and enqueue, never await. */
  onSessionEvent(session: Session, event: SessionEvent): void {
    switch (event.type) {
      case "turn/start": {
        const state = this.ensureSession(session.id);
        state.activeTurn = event.data.turn;
        if (!state.turns.has(event.data.turn)) {
          state.turns.set(event.data.turn, { userText: [], assistantText: [], queued: false });
        }
        return;
      }
      case "user/message": {
        const state = this.sessions.get(session.id);
        if (state?.activeTurn === undefined) return;
        // Only direct human input; our own recall injections (source plugin)
        // and tool results must not re-enter the memory store.
        if (event.data.source.kind !== "user") return;
        const text = textBlocks(event.data.content).trim();
        if (!text) return;
        const turn = state.turns.get(state.activeTurn);
        if (turn && !turn.userText.includes(text)) turn.userText.push(text);
        return;
      }
      case "assistant/message": {
        const state = this.sessions.get(session.id);
        if (state?.activeTurn === undefined) return;
        const message = (event.data as { message?: { content?: unknown } }).message;
        const text = textBlocks(message?.content).trim();
        if (!text) return;
        const turn = state.turns.get(state.activeTurn);
        if (turn && !turn.assistantText.includes(text)) turn.assistantText.push(text);
        return;
      }
      case "turn/end": {
        const state = this.sessions.get(session.id);
        const turn = state?.turns.get(event.data.turn);
        if (!state || !turn || turn.queued) return;
        turn.queued = true;
        if (state.activeTurn === event.data.turn) state.activeTurn = undefined;
        if (turn.userText.length === 0) {
          state.turns.delete(event.data.turn);
          return;
        }
        const messages: Record<string, unknown>[] = [
          { role: "user", content: turn.userText.join("\n\n") },
        ];
        const assistant = turn.assistantText.join("\n\n").trim();
        if (assistant) messages.push({ role: "assistant", content: assistant });
        state.turns.delete(event.data.turn);
        state.dirty = true;
        this.enqueue(session.id, async () => {
          await this.options.client.addMessages(messages, {
            userId: this.options.userId,
            runId: session.id,
            agentId: this.options.agentId || undefined,
            timeoutMs: this.options.captureTimeoutMs,
          });
          this.options.onInfo(
            `capture session=${session.id} turn=${event.data.turn} messages=${messages.length}`,
          );
        });
        return;
      }
      default:
        return;
    }
  }

  /** Detached best-effort flush; never blocks DSH's disposal path. */
  onSessionDisposed(session: Session): void {
    this.flushSession(session.id);
  }

  /**
   * Bounded drain for Cordis disposal. Plugin disposers run before the
   * sessions service emits `session/disposed`, so disposal itself flushes
   * every known dirty session instead of waiting for the event.
   */
  async dispose(drainTimeoutMs = 10_000): Promise<void> {
    for (const sessionId of [...this.sessions.keys()]) {
      this.flushSession(sessionId);
    }
    const deadlineAt = Date.now() + drainTimeoutMs;
    // Re-read each pass: a settled job can have enqueued a successor.
    while (this.pendingBySession.size > 0 && Date.now() < deadlineAt) {
      let timer: ReturnType<typeof setTimeout> | undefined;
      try {
        await Promise.race([
          Promise.allSettled([...this.pendingBySession.values()]),
          new Promise<"timeout">((resolve) => {
            timer = setTimeout(() => resolve("timeout"), Math.max(0, deadlineAt - Date.now()));
          }),
        ]);
      } finally {
        if (timer !== undefined) clearTimeout(timer);
      }
    }
  }

  private flushSession(sessionId: string): void {
    const state = this.sessions.get(sessionId);
    this.sessions.delete(sessionId);
    if (!state || !state.dirty || state.flushed) return;
    state.flushed = true;
    this.enqueue(sessionId, async () => {
      await this.options.client.flush({
        userId: this.options.userId,
        runId: sessionId,
        timeoutMs: this.options.flushTimeoutMs,
      });
      this.options.onInfo(`flush session=${sessionId}`);
    });
  }

  private ensureSession(sessionId: string): SessionCapture {
    let state = this.sessions.get(sessionId);
    if (!state) {
      state = { turns: new Map(), dirty: false, flushed: false };
      this.sessions.set(sessionId, state);
    }
    return state;
  }

  /** Serial per-session chain; job failures are logged and never break it. */
  private enqueue(sessionId: string, job: () => Promise<void>): void {
    const previous = this.pendingBySession.get(sessionId) ?? Promise.resolve();
    const next = previous.then(async () => {
      try {
        await job();
      } catch (error) {
        this.options.onWarn(`capture job failed for session ${sessionId}`, error);
      }
    });
    this.pendingBySession.set(
      sessionId,
      next.finally(() => {
        if (this.pendingBySession.get(sessionId) === next) {
          this.pendingBySession.delete(sessionId);
        }
      }),
    );
  }
}
