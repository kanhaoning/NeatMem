/**
 * Tests for config.ts — mem0ConfigSchema.parse() and exported constants.
 */
import { describe, it, expect } from "vitest";
import {
  mem0ConfigSchema,
  DEFAULT_CUSTOM_INSTRUCTIONS,
  DEFAULT_CUSTOM_CATEGORIES,
} from "../config.ts";

// ---------------------------------------------------------------------------
// Exported constants
// ---------------------------------------------------------------------------
describe("DEFAULT_CUSTOM_INSTRUCTIONS", () => {
  it("is a non-empty string", () => {
    expect(typeof DEFAULT_CUSTOM_INSTRUCTIONS).toBe("string");
    expect(DEFAULT_CUSTOM_INSTRUCTIONS.length).toBeGreaterThan(0);
  });
});

describe("DEFAULT_CUSTOM_CATEGORIES", () => {
  it("is a non-empty object with string values", () => {
    expect(typeof DEFAULT_CUSTOM_CATEGORIES).toBe("object");
    const keys = Object.keys(DEFAULT_CUSTOM_CATEGORIES);
    expect(keys.length).toBeGreaterThan(0);
    for (const key of keys) {
      expect(typeof DEFAULT_CUSTOM_CATEGORIES[key]).toBe("string");
    }
  });
});

// ---------------------------------------------------------------------------
// mem0ConfigSchema.parse() — defaults
// ---------------------------------------------------------------------------
describe("mem0ConfigSchema.parse() — defaults", () => {
  it("baseUrl defaults to local NeatMem server", () => {
    const cfg = mem0ConfigSchema.parse({ apiKey: "test-key" });
    expect(cfg.baseUrl).toBe("http://localhost:8790");
  });

  it("userId falls back to a non-empty string when not provided", () => {
    const cfg = mem0ConfigSchema.parse({ apiKey: "test-key" });
    expect(typeof cfg.userId).toBe("string");
    expect(cfg.userId.length).toBeGreaterThan(0);
  });

  it("autoCapture defaults to true", () => {
    const cfg = mem0ConfigSchema.parse({ apiKey: "test-key" });
    expect(cfg.autoCapture).toBe(true);
  });

  it("autoRecall defaults to true", () => {
    const cfg = mem0ConfigSchema.parse({ apiKey: "test-key" });
    expect(cfg.autoRecall).toBe(true);
  });

  it("topK defaults to 5", () => {
    const cfg = mem0ConfigSchema.parse({ apiKey: "test-key" });
    expect(cfg.topK).toBe(5);
  });

  it("searchThreshold defaults to 0.1", () => {
    const cfg = mem0ConfigSchema.parse({ apiKey: "test-key" });
    expect(cfg.searchThreshold).toBe(0.1);
  });

  it("customInstructions defaults to DEFAULT_CUSTOM_INSTRUCTIONS", () => {
    const cfg = mem0ConfigSchema.parse({ apiKey: "test-key" });
    expect(cfg.customInstructions).toBe(DEFAULT_CUSTOM_INSTRUCTIONS);
  });

  it("customCategories defaults to DEFAULT_CUSTOM_CATEGORIES", () => {
    const cfg = mem0ConfigSchema.parse({ apiKey: "test-key" });
    expect(cfg.customCategories).toBe(DEFAULT_CUSTOM_CATEGORIES);
  });
});

// ---------------------------------------------------------------------------
// mem0ConfigSchema.parse() — userId precedence
// ---------------------------------------------------------------------------
describe("mem0ConfigSchema.parse() — userId", () => {
  it("userId from config takes precedence over os.userInfo() fallback", () => {
    const cfg = mem0ConfigSchema.parse({
      apiKey: "test-key",
      userId: "custom-user",
    });
    expect(cfg.userId).toBe("custom-user");
  });

  it("empty string userId falls back to os.userInfo()", () => {
    const cfg = mem0ConfigSchema.parse({ apiKey: "test-key", userId: "" });
    // Empty string is falsy, so the fallback should kick in
    expect(typeof cfg.userId).toBe("string");
    expect(cfg.userId.length).toBeGreaterThan(0);
  });

  it("non-string userId falls back to os.userInfo()", () => {
    const cfg = mem0ConfigSchema.parse({ apiKey: "test-key", userId: 123 });
    expect(typeof cfg.userId).toBe("string");
    expect(cfg.userId.length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// mem0ConfigSchema.parse() — needsSetup
// ---------------------------------------------------------------------------
describe("mem0ConfigSchema.parse() — needsSetup", () => {
  // Note: needsSetup = !resolvedApiKey.
  // resolvedApiKey can come from the config OR from the openclaw.json plugin
  // section fallback. When no apiKey is provided anywhere, needsSetup is true.

  it("needsSetup is consistent with empty config", () => {
    const cfg = mem0ConfigSchema.parse({});
    if (cfg.apiKey) {
      expect(cfg.needsSetup).toBe(false);
    } else {
      expect(cfg.needsSetup).toBe(true);
    }
  });

  it("is false when apiKey is explicitly provided", () => {
    const cfg = mem0ConfigSchema.parse({ apiKey: "my-api-key" });
    expect(cfg.needsSetup).toBe(false);
  });

  it("needsSetup is always false when apiKey is a valid string", () => {
    const cfg = mem0ConfigSchema.parse({ apiKey: "test-key-123" });
    expect(cfg.apiKey).toBe("test-key-123");
    expect(cfg.needsSetup).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// mem0ConfigSchema.parse() — error cases
// ---------------------------------------------------------------------------
describe("mem0ConfigSchema.parse() — error cases", () => {
  it("throws on unknown keys", () => {
    expect(() =>
      mem0ConfigSchema.parse({ apiKey: "k", unknownKey: "value" }),
    ).toThrow(/unknown keys.*unknownKey/);
  });

  it("throws when multiple unknown keys are present", () => {
    expect(() =>
      mem0ConfigSchema.parse({ apiKey: "k", foo: 1, bar: 2 }),
    ).toThrow(/unknown keys/);
  });

  it("throws on null input", () => {
    expect(() => mem0ConfigSchema.parse(null)).toThrow(
      "openclaw-neatmem config required",
    );
  });

  it("throws on undefined input", () => {
    expect(() => mem0ConfigSchema.parse(undefined)).toThrow(
      "openclaw-neatmem config required",
    );
  });

  it("throws on string input", () => {
    expect(() => mem0ConfigSchema.parse("not an object")).toThrow(
      "openclaw-neatmem config required",
    );
  });

  it("throws on number input", () => {
    expect(() => mem0ConfigSchema.parse(42)).toThrow(
      "openclaw-neatmem config required",
    );
  });

  it("throws on array input", () => {
    expect(() => mem0ConfigSchema.parse([1, 2, 3])).toThrow(
      "openclaw-neatmem config required",
    );
  });

  it("throws on boolean input", () => {
    expect(() => mem0ConfigSchema.parse(true)).toThrow(
      "openclaw-neatmem config required",
    );
  });

  it("throws on the removed 'mode' key (hard-deleted with OSS mode)", () => {
    expect(() =>
      mem0ConfigSchema.parse({ mode: "platform", apiKey: "k" }),
    ).toThrow(/unknown keys.*mode/);
  });

  it("throws on the removed 'oss' key (hard-deleted with OSS mode)", () => {
    expect(() =>
      mem0ConfigSchema.parse({ oss: {}, apiKey: "k" }),
    ).toThrow(/unknown keys.*oss/);
  });

  it("throws on the removed 'customPrompt' key (hard-deleted with OSS mode)", () => {
    expect(() =>
      mem0ConfigSchema.parse({ customPrompt: "x", apiKey: "k" }),
    ).toThrow(/unknown keys.*customPrompt/);
  });
});

// ---------------------------------------------------------------------------
// mem0ConfigSchema.parse() — explicit overrides
// ---------------------------------------------------------------------------
describe("mem0ConfigSchema.parse() — explicit overrides", () => {
  it("autoCapture can be set to false", () => {
    const cfg = mem0ConfigSchema.parse({
      apiKey: "k",
      autoCapture: false,
    });
    expect(cfg.autoCapture).toBe(false);
  });

  it("autoRecall can be set to false", () => {
    const cfg = mem0ConfigSchema.parse({
      apiKey: "k",
      autoRecall: false,
    });
    expect(cfg.autoRecall).toBe(false);
  });

  it("custom topK is used when provided", () => {
    const cfg = mem0ConfigSchema.parse({ apiKey: "k", topK: 20 });
    expect(cfg.topK).toBe(20);
  });

  it("custom searchThreshold is used when provided", () => {
    const cfg = mem0ConfigSchema.parse({
      apiKey: "k",
      searchThreshold: 0.8,
    });
    expect(cfg.searchThreshold).toBe(0.8);
  });

  it("custom customInstructions override defaults", () => {
    const custom = "My custom instructions";
    const cfg = mem0ConfigSchema.parse({
      apiKey: "k",
      customInstructions: custom,
    });
    expect(cfg.customInstructions).toBe(custom);
  });

  it("custom customCategories override defaults", () => {
    const cats = { myCategory: "description" };
    const cfg = mem0ConfigSchema.parse({
      apiKey: "k",
      customCategories: cats,
    });
    expect(cfg.customCategories).toEqual(cats);
  });

  it("baseUrl is passed through when provided", () => {
    const cfg = mem0ConfigSchema.parse({
      apiKey: "k",
      baseUrl: "https://custom.api.com",
    });
    expect(cfg.baseUrl).toBe("https://custom.api.com");
  });

});

// ---------------------------------------------------------------------------
// mem0ConfigSchema.parse() — skills config (removed in 2.0.0)
// ---------------------------------------------------------------------------
describe("mem0ConfigSchema.parse() — skills config (removed)", () => {
  it("throws on the removed 'skills' key", () => {
    expect(() =>
      mem0ConfigSchema.parse({ apiKey: "k", skills: { recall: { threshold: 0.7 } } }),
    ).toThrow(/unknown keys.*skills/);
  });
});

// ---------------------------------------------------------------------------
// mem0ConfigSchema.parse() — recall config
// ---------------------------------------------------------------------------
describe("mem0ConfigSchema.parse() — recall config", () => {
  it("recall defaults to undefined", () => {
    const cfg = mem0ConfigSchema.parse({ apiKey: "k" });
    expect(cfg.recall).toBeUndefined();
  });

  it("parses top-level recall knobs", () => {
    const cfg = mem0ConfigSchema.parse({
      apiKey: "k",
      recall: { threshold: 0.6, rerank: false, keywordSearch: false, filterMemories: true },
    });
    expect(cfg.recall).toEqual({
      threshold: 0.6,
      rerank: false,
      keywordSearch: false,
      filterMemories: true,
    });
  });

  it("drops unknown recall knobs", () => {
    const cfg = mem0ConfigSchema.parse({
      apiKey: "k",
      recall: { threshold: 0.6, tokenBudget: 999 },
    });
    expect(cfg.recall).toEqual({ threshold: 0.6 });
  });

  it("non-object recall is ignored", () => {
    const cfg = mem0ConfigSchema.parse({ apiKey: "k", recall: "invalid" });
    expect(cfg.recall).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// mem0ConfigSchema.parse() — customCategories edge cases
// ---------------------------------------------------------------------------
describe("mem0ConfigSchema.parse() — customCategories edge cases", () => {
  it("non-object customCategories falls back to defaults", () => {
    const cfg = mem0ConfigSchema.parse({
      apiKey: "k",
      customCategories: "not-an-object",
    });
    expect(cfg.customCategories).toBe(DEFAULT_CUSTOM_CATEGORIES);
  });

  it("array customCategories falls back to defaults", () => {
    const cfg = mem0ConfigSchema.parse({
      apiKey: "k",
      customCategories: ["a", "b"],
    });
    expect(cfg.customCategories).toBe(DEFAULT_CUSTOM_CATEGORIES);
  });

  it("null customCategories falls back to defaults", () => {
    const cfg = mem0ConfigSchema.parse({
      apiKey: "k",
      customCategories: null,
    });
    expect(cfg.customCategories).toBe(DEFAULT_CUSTOM_CATEGORIES);
  });
});

// ---------------------------------------------------------------------------
// mem0ConfigSchema.parse() — non-string apiKey
// ---------------------------------------------------------------------------
describe("mem0ConfigSchema.parse() — apiKey edge cases", () => {
  // Note: When a non-string apiKey is provided, the parser treats it as
  // undefined. However, readMem0ConfigFile() may still provide a fallback
  // apiKey from ~/.mem0/config.json if one exists on the system.

  it("non-string apiKey is not used directly from config", () => {
    const cfg = mem0ConfigSchema.parse({ apiKey: 12345 });
    // The numeric value is not used directly — apiKey comes from fallback or is undefined
    // Either way, the non-string value is never the resolved apiKey
    expect(cfg.apiKey).not.toBe(12345);
  });

  it("boolean apiKey is not used directly from config", () => {
    const cfg = mem0ConfigSchema.parse({ apiKey: true });
    expect(cfg.apiKey).not.toBe(true);
  });

  it("string apiKey takes precedence over any fallback", () => {
    const cfg = mem0ConfigSchema.parse({ apiKey: "explicit-key" });
    expect(cfg.apiKey).toBe("explicit-key");
    expect(cfg.needsSetup).toBe(false);
  });
});
