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
import { readText } from "../fs-safe.ts";
import type { PluginAuthConfig } from "./config-file.ts";
import {
  readPluginAuth,
  writePluginAuth,
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
 * Saves api_key, user_id — and base_url when explicitly provided.
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
    ...(userEmail && { userEmail }),
    ...(baseUrl && { baseUrl }),
  });

  console.log(`  Configuration saved to ${OPENCLAW_CONFIG_FILE}`);
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
              const hasExistingConfig = !!existingAuth.apiKey;
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
      // Matches Python CLI key names (snake_case).
      const CONFIG_KEYS: Record<string, string> = {
        // Short aliases (matches Python CLI)
        api_key: "apiKey",
        email: "userEmail",
        base_url: "baseUrl",
        user_id: "userId",
        auto_recall: "autoRecall",
        auto_capture: "autoCapture",
        top_k: "topK",
      };

      // Keys that contain secrets — redact in show/get output
      const SECRET_KEYS = new Set(["apiKey"]);

      // Boolean config fields — coerce "true"/"1"/"yes" on set
      const BOOLEAN_KEYS = new Set([
        "autoRecall",
        "autoCapture",
      ]);

      // Integer config fields — coerce to number on set
      const INTEGER_KEYS = new Set(["topK"]);

      /** Resolve a user-facing key to the internal camelCase field name. */
      function resolveConfigKey(key: string): string | null {
        return CONFIG_KEYS[key] ?? null;
      }

      /** Read a config value by internal field name. */
      function getConfigValue(field: string): unknown {
        const auth = readPluginAuth();
        const values: Record<string, unknown> = {
          apiKey: auth.apiKey ?? cfg.apiKey,
          baseUrl: auth.baseUrl ?? cfg.baseUrl ?? getBaseUrl(),
          userId: auth.userId ?? cfg.userId,
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
          const entries: Array<[string, string]> = [
            ["user_id", "userId"],
            ["auto_recall", "autoRecall"],
            ["auto_capture", "autoCapture"],
            ["top_k", "topK"],
            ["api_key", "apiKey"],
            ["email", "userEmail"],
          ];

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
          console.log("    openclaw neatmem config set auto_recall false");
          console.log("    openclaw neatmem config set top_k 10");
          console.log("");
        });

      configCmd
        .command("get")
        .description("Get a config value")
        .argument("<key>", "Config key (e.g. user_id, api_key)")
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
        .argument("<key>", "Config key (e.g. user_id, api_key)")
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

          writePluginAuth({ [field]: value } as PluginAuthConfig);
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

    },
    {
      descriptors: [
        { name: "neatmem", description: "NeatMem memory plugin commands", hasSubcommands: true },
      ],
    },
  );
}
