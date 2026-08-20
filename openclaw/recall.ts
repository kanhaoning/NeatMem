/**
 * Query sanitization for recall.
 *
 * (The token-budgeted, category-ranked recall engine that used to live here
 * was removed in 2.0.0 along with skills mode.)
 */

/**
 * Strip OpenClaw metadata prefix from event.prompt before using as search query.
 * This only removes framework noise (sender metadata, timestamps) — NOT
 * conversational rewriting. Query rewriting is the agent's responsibility.
 */
export function sanitizeQuery(raw: string): string {
  let cleaned = raw.replace(
    /Sender\s*\(untrusted metadata\):\s*```json[\s\S]*?```\s*/gi,
    "",
  );
  cleaned = cleaned.replace(/^\[.*?\]\s*/g, "");
  cleaned = cleaned.trim();
  return cleaned || raw;
}
