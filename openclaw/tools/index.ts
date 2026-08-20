import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import type { Mem0Config, Mem0Provider, AddOptions, SearchOptions } from "../types.ts";
import type { Backend } from "../backend/base.ts";

import { createMemorySearchTool } from "./memory-search.ts";
import { createMemoryGetTool } from "./memory-get.ts";
import { createMemoryListTool } from "./memory-list.ts";
import { createMemoryUpdateTool } from "./memory-update.ts";
import { createMemoryDeleteTool } from "./memory-delete.ts";

export interface ToolDeps {
  api: OpenClawPluginApi;
  provider: Mem0Provider;
  cfg: Mem0Config;
  backend?: Backend;
  resolveUserId: (opts: { agentId?: string; userId?: string }) => string;
  effectiveUserId: (sessionKey?: string) => string;
  agentUserId: (id: string) => string;
  buildAddOptions: (userIdOverride?: string, runId?: string, sessionKey?: string) => AddOptions;
  buildSearchOptions: (userIdOverride?: string, limit?: number, runId?: string, sessionKey?: string) => SearchOptions;
  getCurrentSessionId: () => string | undefined;
}

// Write scheduling is pipeline-driven (auto-capture). memory_add stays
// implemented (memory-add.ts) but is deliberately NOT registered: with
// auto-capture on, a manual add landing before batch extraction suppresses
// pipeline extraction via dedup (hermes experiment, 2026-08-17).
// memory_event_list/status were removed in 2.0.0 — the local server's event
// endpoints are permanent empty stubs (main.py "兼容平台模式").
export function registerAllTools(deps: ToolDeps): void {
  const { api } = deps;

  api.registerTool(createMemorySearchTool(deps));
  api.registerTool(createMemoryGetTool(deps));
  api.registerTool(createMemoryListTool(deps));
  api.registerTool(createMemoryUpdateTool(deps));
  api.registerTool(createMemoryDeleteTool(deps));
}
