/**
 * CLI subcommand registration for the OpenClaw NeatMem plugin.
 *
 * Registers all `openclaw neatmem <subcommand>` commands:
 *
 * Memory:
 *   - add         : Add a memory from text (--user-id, --agent-id)
 *   - search      : Search memories (--top-k, --scope, --agent-id, --user-id)
 *   - get         : Get a specific memory by ID
 *   - list        : List memories with optional filters (--user-id, --agent-id, --top-k)
 *   - update      : Update a memory's text
 *   - delete      : Delete a memory or all memories (--all, --confirm)
 *   - import      : Import memories from a JSON file
 *
 * Management:
 *   - init        : Authenticate with NeatMem Platform (email or API key)
 *   - status      : Check API connectivity and show current config
 *   - config show : Display current plugin configuration
 *   - config get  : Get a single config value
 *   - config set  : Update a plugin config field
 *   - event list  : List recent background events
 *   - event status: Get status of a specific event
 *   - dream       : Run memory consolidation
 *
 * Naming conventions match the Python CLI (`neatmem init`, `neatmem search`, etc.)
 */

import { createInterface } from "node:readline";
import { userInfo as osUserInfo } from "node:os";

import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import type { Backend } from "../backend/base.ts";
import type {
  Mem0Config,
  Mem0Provider,
  MemoryItem,
  SearchOptions,
} from "../types.ts";
import { loadDreamPrompt } from "../skill-loader.ts";
import { readText } from "../fs-safe.ts";
import type { PluginAuthConfig } from "./config-file.ts";
import {
  readPluginAuth,
  writePluginAuth,
  writePluginConfigField,
  getBaseUrl,
  OPENCLAW_CONFIG_FILE,
} from "./config-file.ts";

// ============================================================================
// Reusable helpers (DRY)
// ============================================================================

/** Prompt user for input on stderr (keeps stdout clean for piping). */
function promptInput(question: string, prefill?: string): Promise<string> {
  const rl = createInterface({ input: process.stdin, output: process.stderr });
  return new Promise((resolve) => {
    rl.question(question, (answer) => {
      rl.close();
      resolve(answer.trim());
    });
    if (prefill) rl.write(prefill);
  });
}

/** Get system username for userId fallback. */
function getSystemUsername(): string {
  try {
    return osUserInfo().username || "default";
  } catch {
    return "default";
  }
}

/**
 * Resolve userId silently (no interactive prompt).
 * Matches Python CLI: --user-id flag > existing config > system username > "default"
 * Uses os.userInfo().username which covers all platforms.
 */
function resolveUserId(flagValue?: string, existingValue?: string): string {
  if (flagValue) return flagValue;
  if (existingValue) return existingValue;
  return getSystemUsername();
}

/** Validate an API key by pinging the platform. Returns true if valid. */
async function validateApiKey(
  baseUrl: string,
  apiKey: string,
): Promise<{ ok: boolean; status?: number; error?: string; userEmail?: string }> {
  try {
    const resp = await fetch(`${baseUrl}/v1/ping/`, {
      headers: {
        Authorization: `Token ${apiKey}`,
        "X-NeatMem-Source": "OPENCLAW",
        "X-NeatMem-Client-Language": "node",
      },
    });
    if (!resp.ok) return { ok: false, status: resp.status };
    try {
      const data = (await resp.json()) as Record<string, unknown>;
      return { ok: true, userEmail: data.user_email as string | undefined };
    } catch {
      return { ok: true };
    }
  } catch (err) {
    return { ok: false, error: String(err) };
  }
}

/**
 * Save login config and print summary.
 * Saves api_key, user_id, mode — and base_url when explicitly provided.
 */
function saveLoginConfig(
  apiKey: string,
  userIdFlag?: string,
  userEmail?: string,
  baseUrl?: string,
): void {
  const existingAuth = readPluginAuth();
  const userId = resolveUserId(userIdFlag, existingAuth.userId);

  writePluginAuth({
    apiKey,
    userId,
    mode: "platform",
    ...(userEmail && { userEmail }),
    ...(baseUrl && { baseUrl }),
  });

  console.log(`  Configuration saved to ${OPENCLAW_CONFIG_FILE}`);
  console.log(`  Mode: platform`);
  console.log(`  User ID: ${userId}`);
}

// ============================================================================
// Main registration function
// ============================================================================

export function registerCliCommands(
  api: OpenClawPluginApi,
  backend: Backend,
  provider: Mem0Provider,
  cfg: Mem0Config,
  effectiveUserId: (sessionKey?: string) => string,
  agentUserId: (id: string) => string,
  buildSearchOptions: (
    userIdOverride?: string,
    limit?: number,
    runId?: string,
    sessionKey?: string,
  ) => SearchOptions,
  getCurrentSessionId: () => string | undefined,
): void {
  api.registerCli(
    ({ program }) => {
      const neatmem = program
        .command("neatmem")
        .description("NeatMem memory plugin commands")
        .configureHelp({ sortSubcommands: false, subcommandTerm: (cmd) => cmd.name() });

      // ====================================================================
      // init (matches: neatmem init)
      // ====================================================================

      neatmem
        .command("init")
        .description("Set up NeatMem — write config and validate (works with zero flags)")
        .option("--api-key <key>", "API token (any value works for a local NeatMem server)")
        .option("--user-id <id>", "Set user ID for memory namespace")
        .option("--base-url <url>", "NeatMem server URL (default: http://localhost:8790)")
        .action(
          async (opts: {
            apiKey?: string;
            userId?: string;
            baseUrl?: string;
          }) => {
            try {
              const baseUrl = opts.baseUrl ?? getBaseUrl();
              const existingAuth = readPluginAuth();
              const hasExistingConfig = !!(existingAuth.apiKey || existingAuth.mode);
              const apiKey = opts.apiKey ?? existingAuth.apiKey ?? "neatmem-local";

              const check = await validateApiKey(baseUrl, apiKey);
              saveLoginConfig(apiKey, opts.userId, check.userEmail, baseUrl);

              if (hasExistingConfig) {
                console.log(
                  "  Existing configuration detected — updated provided fields (other settings preserved).",
                );
              }

              if (check.ok) {
                console.log("  API key validated. Connected to NeatMem.");
              } else if (check.status) {
                console.warn(
                  `  Config saved but validation returned HTTP ${check.status}. Check that the server is healthy.`,
                );
              } else {
                console.warn(
                  `  Config saved but could not reach ${baseUrl}: ${check.error}. Start the NeatMem server and re-run init to validate.`,
                );
              }
              console.log(
                "  Restart the gateway: openclaw gateway restart\n",
              );
            } catch (err) {
              console.error(`Init failed: ${String(err)}`);
            }
          },
        );

      // ====================================================================
      // search (matches: neatmem search <query> --top-k --user-id --agent-id)
      // ====================================================================

      neatmem
        .command("search")
        .description("Search memories")
        .argument("<query>", "Search query")
        .option("--top-k <n>", "Max results", String(cfg.topK))
        .option(
          "--scope <scope>",
          'Memory scope: "session", "long-term", or "all"',
          "all",
        )
        .option("--agent-id <agentId>", "Search agent's memory namespace")
        .option("--user-id <userId>", "Override user ID")
        .action(
          async (
            query: string,
            opts: {
              topK: string;
              scope: string;
              agentId?: string;
              userId?: string;
            },
          ) => {
            try {
              const limit = parseInt(opts.topK, 10);
              const scope = opts.scope as "session" | "long-term" | "all";
              const currentSessionId = getCurrentSessionId();
              const uid = opts.userId
                ? opts.userId
                : opts.agentId
                  ? agentUserId(opts.agentId)
                  : effectiveUserId(currentSessionId);

              // CLI search: no source filter so users find ALL memories
              const cliSearchOpts = (
                userIdOverride?: string,
                lim?: number,
                runId?: string,
              ): SearchOptions => {
                const base = buildSearchOptions(userIdOverride, lim, runId);
                base.threshold = 0.3;
                return base;
              };

              let allResults: MemoryItem[] = [];

              if (scope === "session" || scope === "all") {
                if (currentSessionId) {
                  const sessionResults = await provider.search(
                    query,
                    cliSearchOpts(uid, limit, currentSessionId),
                  );
                  if (sessionResults?.length) {
                    allResults.push(
                      ...sessionResults.map((r) => ({
                        ...r,
                        _scope: "session" as const,
                      })),
                    );
                  }
                } else if (scope === "session") {
                  console.log(
                    "No active session ID available for session-scoped search.",
                  );
                  return;
                }
              }

              if (scope === "long-term" || scope === "all") {
                const longTermResults = await provider.search(
                  query,
                  cliSearchOpts(uid, limit),
                );
                if (longTermResults?.length) {
                  allResults.push(
                    ...longTermResults.map((r) => ({
                      ...r,
                      _scope: "long-term" as const,
                    })),
                  );
                }
              }

              // Deduplicate by ID when searching "all"
              if (scope === "all") {
                const seen = new Set<string>();
                allResults = allResults.filter((r) => {
                  if (seen.has(r.id)) return false;
                  seen.add(r.id);
                  return true;
                });
              }

              if (!allResults.length) {
                console.log("No memories found.");
                return;
              }

              const output = allResults.map((r) => ({
                id: r.id,
                memory: r.memory,
                score: r.score,
                scope: (r as any)._scope,
                categories: r.categories,
                created_at: r.created_at,
              }));
              console.log(JSON.stringify(output, null, 2));
            } catch (err) {
              console.error(`Search failed: ${String(err)}`);
            }
          },
        );

      // ====================================================================
      // add (matches: neatmem add <text> --user-id --agent-id)
      // ====================================================================

      neatmem
        .command("add")
        .description("Add a memory from text")
        .argument("<text>", "Text to store as a memory")
        .option("--user-id <userId>", "Override user ID")
        .option("--agent-id <agentId>", "Store in agent's memory namespace")
        .action(
          async (
            text: string,
            opts: { userId?: string; agentId?: string },
          ) => {
            try {
              const uid = opts.userId
                ? opts.userId
                : opts.agentId
                  ? agentUserId(opts.agentId)
                  : effectiveUserId(getCurrentSessionId());
              const result = await provider.add(
                [{ role: "user", content: text }],
                { user_id: uid, source: "OPENCLAW" },
              );
              const count = result.results?.length ?? 0;
              if (count > 0) {
                console.log(`Added ${count} memory(s):`);
                for (const r of result.results) {
                  console.log(`  ${r.id}: ${r.memory} [${r.event}]`);
                }
              } else {
                console.log(
                  "No new memories extracted (text may already be stored or not contain durable facts).",
                );
              }
            } catch (err) {
              console.error(`Add failed: ${String(err)}`);
            }
          },
        );

      // ====================================================================
      // get (matches: neatmem get <memory_id>)
      // ====================================================================

      neatmem
        .command("get")
        .description("Get a specific memory by ID")
        .argument("<memory_id>", "Memory ID to retrieve")
        .action(async (memoryId: string) => {
          try {
            const memory = await provider.get(memoryId);
            console.log(
              JSON.stringify(
                {
                  id: memory.id,
                  memory: memory.memory,
                  user_id: memory.user_id,
                  categories: memory.categories,
                  metadata: memory.metadata,
                  created_at: memory.created_at,
                  updated_at: memory.updated_at,
                },
                null,
                2,
              ),
            );
          } catch (err) {
            console.error(`Get failed: ${String(err)}`);
          }
        });

      // ====================================================================
      // list (matches: neatmem list --user-id --agent-id --top-k)
      // ====================================================================

      neatmem
        .command("list")
        .description("List memories with optional filters")
        .option("--user-id <userId>", "Override user ID")
        .option("--agent-id <agentId>", "List agent's memories")
        .option("--top-k <n>", "Max results", "50")
        .action(
          async (opts: {
            userId?: string;
            agentId?: string;
            topK: string;
          }) => {
            try {
              const uid = opts.userId
                ? opts.userId
                : opts.agentId
                  ? agentUserId(opts.agentId)
                  : cfg.userId;
              const limit = parseInt(opts.topK, 10);
              const memories = await provider.getAll({
                user_id: uid,
                page_size: limit,
                source: "OPENCLAW",
              });

              if (!Array.isArray(memories) || memories.length === 0) {
                console.log("No memories found.");
                return;
              }

              const output = memories.map((m) => ({
                id: m.id,
                memory: m.memory,
                categories: m.categories,
                created_at: m.created_at,
                updated_at: m.updated_at,
              }));
              console.log(JSON.stringify(output, null, 2));
              console.log(`\nTotal: ${memories.length} memories`);
            } catch (err) {
              console.error(`List failed: ${String(err)}`);
            }
          },
        );

      // ====================================================================
      // update (matches: neatmem update <memory_id> <text>)
      // ====================================================================

      neatmem
        .command("update")
        .description("Update a memory's text")
        .argument("<memory_id>", "Memory ID to update")
        .argument("<text>", "New text for the memory")
        .action(async (memoryId: string, text: string) => {
          try {
            await provider.update(memoryId, text);
            console.log(`Memory ${memoryId} updated.`);
          } catch (err) {
            console.error(`Update failed: ${String(err)}`);
          }
        });

      // ====================================================================
      // delete (matches: neatmem delete <memory_id> --all --user-id)
      // ====================================================================

      neatmem
        .command("delete")
        .description("Delete a memory, or all memories for a user")
        .argument("[memory_id]", "Memory ID to delete")
        .option("--all", "Delete all memories for the user")
        .option("--user-id <userId>", "Override user ID (with --all)")
        .option("--agent-id <agentId>", "Delete from agent's namespace")
        .option("--confirm", "Skip confirmation for bulk delete")
        .action(
          async (
            memoryId: string | undefined,
            opts: {
              all?: boolean;
              userId?: string;
              agentId?: string;
              confirm?: boolean;
            },
          ) => {
            try {
              if (opts.all) {
                const uid = opts.userId
                  ? opts.userId
                  : opts.agentId
                    ? agentUserId(opts.agentId)
                    : cfg.userId;

                if (!opts.confirm && process.stdin.isTTY) {
                  const answer = await promptInput(
                    `  Delete ALL memories for user "${uid}"? This cannot be undone. (yes/N): `,
                  );
                  if (answer.toLowerCase() !== "yes") {
                    console.log("Cancelled.");
                    return;
                  }
                } else if (!opts.confirm) {
                  console.error(
                    "Bulk delete requires --confirm flag in non-interactive mode.",
                  );
                  return;
                }

                await provider.deleteAll(uid);
                console.log(`All memories deleted for user "${uid}".`);
                return;
              }

              if (!memoryId) {
                console.error(
                  "Provide a memory_id or use --all to delete all memories.",
                );
                return;
              }

              await provider.delete(memoryId);
              console.log(`Memory ${memoryId} deleted.`);
            } catch (err) {
              console.error(`Delete failed: ${String(err)}`);
            }
          },
        );

      // ====================================================================
      // status (matches: neatmem status)
      // ====================================================================

      neatmem
        .command("status")
        .description("Check API connectivity and current config")
        .action(async () => {
          try {
            const auth = readPluginAuth();
            console.log(`Mode: ${cfg.mode}`);
            console.log(`User ID: ${cfg.userId}`);
            console.log(`Config: ${OPENCLAW_CONFIG_FILE}`);
            console.log("");

            if (!backend) {
              console.log("Not configured. Run: openclaw neatmem init");
              return;
            }

            const result = await backend.status();
            if (result.connected) {
              console.log("Connected to NeatMem");
            } else {
              console.log("Not connected to NeatMem");
            }
            if (result.url) {
              console.log(`URL: ${String(result.url)}`);
            }
            if (result.error) {
              console.log(`Error: ${String(result.error)}`);
            }
          } catch (err) {
            console.error(`Status check failed: ${String(err)}`);
          }
        });

      // ====================================================================
      // config (matches: neatmem config show, neatmem config get, neatmem config set)
      // ====================================================================

      const configCmd = neatmem
        .command("config")
        .description("Manage plugin configuration");

      // All settable config keys: short alias → camelCase field in openclaw.json
      // Matches Python CLI key names (snake_case) with dot-notation support.
      const CONFIG_KEYS: Record<string, string> = {
        // Short aliases (matches Python CLI)
        api_key: "apiKey",
        email: "userEmail",
        base_url: "baseUrl",
        user_id: "userId",
        auto_recall: "autoRecall",
        auto_capture: "autoCapture",
        top_k: "topK",
        mode: "mode",
        embedder_provider: "oss.embedder.provider",
        embedder_model: "oss.embedder.config.model",
        embedder_key: "oss.embedder.config.apiKey",
        llm_provider: "oss.llm.provider",
        llm_model: "oss.llm.config.model",
        llm_key: "oss.llm.config.apiKey",
        vector_provider: "oss.vectorStore.provider",
        vector_host: "oss.vectorStore.config.host",
        vector_port: "oss.vectorStore.config.port",
        collection_name: "oss.vectorStore.config.collectionName",
        vector_db_name: "oss.vectorStore.config.dbname",
        vector_db_user: "oss.vectorStore.config.user",
        vector_db_path: "oss.vectorStore.config.dbPath",
        history_db_path: "oss.historyDbPath",
        disable_history: "oss.disableHistory",
      };

      // Keys that contain secrets — redact in show/get output
      const SECRET_KEYS = new Set(["apiKey", "oss.embedder.config.apiKey", "oss.llm.config.apiKey"]);

      // Boolean config fields — coerce "true"/"1"/"yes" on set
      const BOOLEAN_KEYS = new Set([
        "autoRecall",
        "autoCapture",
        "oss.disableHistory",
      ]);

      // Integer config fields — coerce to number on set
      const INTEGER_KEYS = new Set(["topK", "oss.vectorStore.config.port"]);

      /** Resolve a user-facing key to the internal camelCase field name. */
      function resolveConfigKey(key: string): string | null {
        return CONFIG_KEYS[key] ?? null;
      }

      /** Read a config value by internal field name. */
      function getConfigValue(field: string): unknown {
        if (field.startsWith("oss.")) {
          const parts = field.split(".");
          let current: unknown = cfg.oss;
          for (let i = 1; i < parts.length && current != null; i++) {
            current = (current as Record<string, unknown>)[parts[i]];
          }
          return current;
        }

        const auth = readPluginAuth();
        const values: Record<string, unknown> = {
          apiKey: auth.apiKey ?? cfg.apiKey,
          baseUrl: auth.baseUrl ?? cfg.baseUrl ?? getBaseUrl(),
          userId: auth.userId ?? cfg.userId,
          mode: auth.mode ?? cfg.mode,
          userEmail: auth.userEmail,
          autoRecall: cfg.autoRecall,
          autoCapture: cfg.autoCapture,
          topK: cfg.topK,
        };
        return values[field];
      }

      /** Redact a secret value for display: first 4 + ... + last 4 */
      function redact(value: string): string {
        if (value.length <= 8) return value.slice(0, 2) + "***";
        return value.slice(0, 4) + "..." + value.slice(-4);
      }

      /** Format a config value for display (redacts secrets). */
      function displayValue(field: string, value: unknown): string {
        if (value === undefined || value === null || value === "") {
          return "(not set)";
        }
        if (SECRET_KEYS.has(field) && typeof value === "string") {
          return redact(value);
        }
        return String(value);
      }

      configCmd
        .command("show")
        .description("Show current configuration")
        .action(() => {
          // Display order: general first, then mode-specific
          const entries: Array<[string, string]> = [
            ["mode", "mode"],
            ["user_id", "userId"],
            ["auto_recall", "autoRecall"],
            ["auto_capture", "autoCapture"],
            ["top_k", "topK"],
          ];

          if (cfg.mode === "platform") {
            entries.push(
              ["api_key", "apiKey"],
              ["email", "userEmail"],
            );
          } else {
            entries.push(
              ["embedder_provider", "oss.embedder.provider"],
              ["embedder_model", "oss.embedder.config.model"],
              ["embedder_key", "oss.embedder.config.apiKey"],
              ["llm_provider", "oss.llm.provider"],
              ["llm_model", "oss.llm.config.model"],
              ["llm_key", "oss.llm.config.apiKey"],
              ["vector_provider", "oss.vectorStore.provider"],
              ["history_db_path", "oss.historyDbPath"],
              ["disable_history", "oss.disableHistory"],
            );
          }

          // Calculate column widths
          const maxKeyLen = Math.max(
            ...entries.map(([k]) => k.length),
            3,
          );

          console.log("");
          console.log(
            `  ${"Key".padEnd(maxKeyLen)}   Value`,
          );
          console.log(
            `  ${"─".repeat(maxKeyLen)}   ${"─".repeat(30)}`,
          );
          for (const [displayKey, field] of entries) {
            const value = getConfigValue(field);
            const display = displayValue(field, value);
            console.log(
              `  ${displayKey.padEnd(maxKeyLen)}   ${display}`,
            );
          }
          console.log("");
          console.log(`  Config file: ${OPENCLAW_CONFIG_FILE}`);
          console.log("");
          console.log("  To change a setting:");
          console.log("    openclaw neatmem config set <key> <value>");
          console.log("");
          console.log("  Examples:");
          if (cfg.mode === "platform") {
            console.log("    openclaw neatmem config set mode open-source");
            console.log("    openclaw neatmem config set auto_recall false");
          } else {
            console.log("    openclaw neatmem config set vector_provider qdrant");
            console.log("    openclaw neatmem config set llm_model gpt-4o");
            console.log("    openclaw neatmem config set embedder_provider openai");
          }
          console.log("");
        });

      configCmd
        .command("get")
        .description("Get a config value")
        .argument("<key>", "Config key (e.g. user_id, api_key, llm_model)")
        .action((key: string) => {
          const field = resolveConfigKey(key);
          if (!field) {
            console.error(
              `Unknown config key: ${key}`,
            );
            return;
          }
          const value = getConfigValue(field);
          console.log(displayValue(field, value));
        });

      configCmd
        .command("set")
        .description("Set a config value")
        .argument("<key>", "Config key (e.g. user_id, api_key, llm_model)")
        .argument("<value>", "New value")
        .action((key: string, rawValue: string) => {
          const field = resolveConfigKey(key);
          if (!field) {
            console.error(
              `Unknown config key: ${key}`,
            );
            return;
          }

          // Type coercion (matches Python CLI behavior)
          let value: unknown = rawValue;
          if (BOOLEAN_KEYS.has(field)) {
            value =
              rawValue.toLowerCase() === "true" ||
              rawValue === "1" ||
              rawValue.toLowerCase() === "yes";
          } else if (INTEGER_KEYS.has(field)) {
            const parsed = parseInt(rawValue, 10);
            if (isNaN(parsed)) {
              console.error(`Invalid integer value: ${rawValue}`);
              return;
            }
            value = parsed;
          }

          // Nested OSS fields use dot-path writer; flat fields use auth writer
          if (field.startsWith("oss.")) {
            writePluginConfigField(field.split("."), value);
          } else {
            writePluginAuth({ [field]: value } as PluginAuthConfig);
          }
          console.log(
            `${key} = ${displayValue(field, value)}`,
          );
        });

      // ====================================================================
      // import (matches: neatmem import <file>)
      // ====================================================================

      neatmem
        .command("import")
        .description("Import memories from a JSON file")
        .argument("<file>", "Path to JSON file containing memories")
        .option("--user-id <userId>", "Override user ID for all imported memories")
        .option("--agent-id <agentId>", "Override agent ID for all imported memories")
        .action(
          async (
            file: string,
            opts: { userId?: string; agentId?: string },
          ) => {
            try {
              let data: unknown;
              try {
                data = JSON.parse(readText(file));
              } catch (err) {
                console.error(`Failed to read file: ${String(err)}`);
                return;
              }

              const items = Array.isArray(data) ? data : [data];
              let added = 0;
              let failed = 0;

              for (const item of items) {
                const content =
                  item?.memory ?? item?.text ?? item?.content ?? "";
                if (!content) {
                  failed++;
                  continue;
                }
                try {
                  await backend.add(content, undefined, {
                    userId: opts.userId ?? item?.user_id ?? cfg.userId,
                    agentId: opts.agentId ?? item?.agent_id,
                    metadata: item?.metadata,
                  });
                  added++;
                } catch {
                  failed++;
                }
              }

              console.log(`Imported ${added} memories.`);
              if (failed) {
                console.error(`${failed} memories failed to import.`);
              }
            } catch (err) {
              console.error(`Import failed: ${String(err)}`);
            }
          },
        );

      // ====================================================================
      // event (matches: neatmem event list, neatmem event status <id>)
      // ====================================================================

      const eventCmd = neatmem
        .command("event")
        .description("Manage background processing events");

      eventCmd
        .command("list")
        .description("List recent background events")
        .action(async () => {
          try {
            if (!backend || cfg.mode === "open-source") {
              console.log("Event tracking is only available in platform mode.");
              return;
            }
            const results = await backend.listEvents();
            if (!results.length) {
              console.log("No events found.");
              return;
            }

            // Table header
            const header = [
              "Event ID".padEnd(36),
              "Type".padEnd(14),
              "Status".padEnd(12),
              "Latency".padStart(10),
              "Created".padEnd(20),
            ].join("  ");
            console.log(header);
            console.log("-".repeat(header.length));

            for (const ev of results) {
              const evId = String(ev.id ?? "");
              const evType = String(ev.event_type ?? "—").padEnd(14);
              const status = String(ev.status ?? "—").padEnd(12);
              const latency = typeof ev.latency === "number"
                ? `${Math.round(ev.latency as number)}ms`
                : "—";
              const created = String(ev.created_at ?? "—")
                .slice(0, 19)
                .replace("T", " ");

              console.log(
                `${evId.padEnd(36)}  ${evType}  ${status}  ${latency.padStart(10)}  ${created}`,
              );
            }
            console.log(`\n${results.length} event${results.length !== 1 ? "s" : ""}`);
          } catch (err) {
            console.error(`Failed to list events: ${String(err)}`);
          }
        });

      eventCmd
        .command("status")
        .description("Get status of a specific background event")
        .argument("<event_id>", "Event ID to check")
        .action(async (eventId: string) => {
          try {
            if (!backend || cfg.mode === "open-source") {
              console.log("Event tracking is only available in platform mode.");
              return;
            }
            const ev = await backend.getEvent(eventId);

            const status = String(ev.status ?? "—");
            const evType = String(ev.event_type ?? "—");
            const latency = typeof ev.latency === "number"
              ? `${Math.round(ev.latency as number)}ms`
              : "—";
            const created = String(ev.created_at ?? "—")
              .slice(0, 19)
              .replace("T", " ");
            const updated = String(ev.updated_at ?? "—")
              .slice(0, 19)
              .replace("T", " ");

            console.log(`Event ID:  ${eventId}`);
            console.log(`Type:      ${evType}`);
            console.log(`Status:    ${status}`);
            console.log(`Latency:   ${latency}`);
            console.log(`Created:   ${created}`);
            console.log(`Updated:   ${updated}`);

            const results = ev.results as Record<string, unknown>[] | undefined;
            if (results && Array.isArray(results) && results.length) {
              console.log(`\nResults (${results.length}):`);
              for (const r of results) {
                const memId = String(r.id ?? "").slice(0, 8);
                const data = r.data as Record<string, unknown> | undefined;
                const memory = data?.memory ?? "";
                const evName = String(r.event ?? "");
                const user = String(r.user_id ?? "");
                let detail = `${evName}  ${memory}`;
                if (user) detail += `  (user_id=${user})`;
                console.log(`  · ${detail}  (${memId})`);
              }
            }
          } catch (err) {
            console.error(`Failed to get event: ${String(err)}`);
          }
        });

      // ====================================================================
      // help (matches: neatmem help, neatmem help --json)
      // ====================================================================

      neatmem
        .command("help")
        .description("Show help. Use --json for machine-readable output (for LLM agents)")
        .option("--json", "Output as JSON for agent/programmatic use")
        .action((opts: { json?: boolean }) => {
          const commands = {
            memory: {
              search: "Query your memory store — semantic, keyword, or hybrid retrieval",
              add: "Add a memory from text, messages, or stdin",
              get: "Get a specific memory by ID",
              list: "List memories with optional filters",
              update: "Update a memory's text or metadata",
              delete: "Delete a memory, all memories, or an entity",
              import: "Import memories from a JSON file",
            },
            management: {
              init: "Interactive setup wizard for neatmem CLI",
              status: "Check connectivity and authentication",
              config: "Manage neatmem configuration (show, get, set)",
              event: "Manage background processing events (list, status)",
              dream: "Run memory consolidation (review, merge, prune)",
              help: "Show help. Use --json for machine-readable output (for LLM agents)",
            },
          };

          if (opts.json) {
            console.log(JSON.stringify({ commands }, null, 2));
            return;
          }

          console.log("");
          console.log("  openclaw neatmem <command>");
          console.log("");
          console.log("  Memory:");
          for (const [cmd, desc] of Object.entries(commands.memory)) {
            console.log(`    ${cmd.padEnd(12)} ${desc}`);
          }
          console.log("");
          console.log("  Management:");
          for (const [cmd, desc] of Object.entries(commands.management)) {
            console.log(`    ${cmd.padEnd(12)} ${desc}`);
          }
          console.log("");
        });

      // ====================================================================
      // dream
      // ====================================================================

      neatmem
        .command("dream")
        .description(
          "Run memory consolidation (review, merge, prune stored memories)",
        )
        .option(
          "--dry-run",
          "Show memory inventory without running consolidation",
        )
        .action(async (opts: { dryRun?: boolean }) => {
          try {
            const uid = cfg.userId;
            const memories = await provider.getAll({
              user_id: uid,
              source: "OPENCLAW",
            });
            const count = Array.isArray(memories) ? memories.length : 0;

            if (count === 0) {
              console.log("No memories to consolidate.");
              return;
            }

            const catCounts = new Map<string, number>();
            for (const mem of memories) {
              const cat =
                (mem.metadata as any)?.category ??
                mem.categories?.[0] ??
                "uncategorized";
              catCounts.set(cat, (catCounts.get(cat) ?? 0) + 1);
            }
            process.stderr.write(`\nMemory inventory for "${uid}":\n`);
            for (const [cat, num] of [...catCounts.entries()].sort(
              (a, b) => b[1] - a[1],
            )) {
              process.stderr.write(`  ${cat}: ${num}\n`);
            }
            process.stderr.write(`  TOTAL: ${count}\n\n`);

            if (opts.dryRun) {
              process.stderr.write("Dry run — no changes made.\n");
              return;
            }

            const dreamPrompt = loadDreamPrompt(cfg.skills ?? {});
            if (!dreamPrompt) {
              process.stderr.write(
                "Dream skill file not found at skills/memory-dream/SKILL.md\n",
              );
              return;
            }

            const memoryDump = (memories as MemoryItem[])
              .map((m, i) => {
                const cat =
                  (m.metadata as any)?.category ??
                  m.categories?.[0] ??
                  "uncategorized";
                const imp = (m.metadata as any)?.importance ?? "?";
                const created = m.created_at ?? "unknown";
                return `${i + 1}. [${m.id}] (${cat}, importance: ${imp}, created: ${created}) ${m.memory}`;
              })
              .join("\n");

            const fullPrompt = [
              "<dream-protocol>",
              dreamPrompt,
              "</dream-protocol>",
              "",
              `<all-memories count="${count}" user="${uid}">`,
              memoryDump,
              "</all-memories>",
              "",
              "Begin consolidation. Review all memories above and execute merge, delete, and rewrite operations using the available tools.",
            ].join("\n");

            process.stdout.write(fullPrompt + "\n");
            process.stderr.write(
              `Dream prompt written to stdout (${fullPrompt.length} chars). Paste it into an OpenClaw session to run consolidation.\n`,
            );
          } catch (err) {
            console.error(`Dream failed: ${String(err)}`);
          }
        });
    },
    {
      descriptors: [
        { name: "neatmem", description: "NeatMem memory plugin commands", hasSubcommands: true },
      ],
    },
  );
}
