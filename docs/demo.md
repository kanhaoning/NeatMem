# Demo

`neatmem demo` runs a small scenario through the memory pipeline and prints
every decision along the way: what gets extracted from each message, how the
deduplication step judges it against existing memories (add / update / skip,
with the reason), and the final contents of the store.

Use it to see how NeatMem behaves on your own examples before wiring it into
an agent. Each run uses a fresh in-memory database — nothing is written to
disk and no state carries over between runs.

## Quick example

```bash
neatmem demo \
  --say "I live in Beijing, Haidian district. I've been there for over three years." \
  --say "I moved to Shanghai last week, renting an apartment in Xuhui."
```

Output (abridged; pipeline log lines omitted):

```text
Plan: 0 existing memories (verbatim), 2 session(s) replayed in order, reps=1
Server: internal, 127.0.0.1:44431 (memory)

-- session s1 turn 1/1 --
  + add: User lives in Beijing's Haidian district, has been residing there for
    over three years ...

-- session s2 turn 1/1 --
  ~ update (update_rewrite, score=0.7426): "User lives in Beijing's Haidian
    district ..." -> "User relocated from Beijing's Haidian district to
    Shanghai's Xuhui district ..."

Final store (1 memories):
  [1] User relocated from Beijing's Haidian district to Shanghai's Xuhui
      district, renting an apartment there ... after residing in Haidian for
      over three years.
```

The second message contradicts the stored fact, so instead of adding a
duplicate, the store is updated in place.

- `--say TEXT` sends one user message per value, in order (repeatable). Each
  `--say` is a separate write, so later messages are deduplicated against what
  earlier ones stored.
- `--existing TEXT` seeds a memory verbatim before the replay starts
  (repeatable).

Because deduplication involves LLM judgment, a single run is stochastic. Use
`--reps` to repeat the whole scenario from a fresh store and get a summary:

```bash
neatmem demo --say "..." --say "..." --reps 5
```

## Prerequisites

Same provider configuration as `neatmem serve` (LLM + embedding API keys, via
environment variables or `.env`). See [Configuration](configuration.md). No
qdrant server or database setup is needed.

## Case files

For anything beyond inline messages — multi-session scenarios, assistant
messages, seeded memories with timestamps — describe the scenario in a JSON
file and pass its path:

```bash
neatmem demo my-case.json
# or via stdin:
cat my-case.json | neatmem demo -
```

```json
{
  "name": "residence move",
  "description": "What happens when a later message contradicts a stored fact.",
  "existing_memories": [
    {"text": "The user lives in Beijing, Haidian district.", "created_at": "2026-07-01"}
  ],
  "sessions": [
    {"messages": [{"role": "user", "content": "I moved to Shanghai last week..."}]},
    {"messages": [{"role": "user", "content": "My new commute is much shorter."}]}
  ]
}
```

- `existing_memories` are written verbatim before the replay starts.
- `sessions` are replayed in order, one write call per session.

## Tuning the behavior

`neatmem demo` accepts the same flags as `neatmem serve`, so you can watch how
a setting changes the outcome. Deduplication is on by default; some examples:

```bash
# Turn dedup off — conflicting facts pile up as separate memories
neatmem demo --say "..." --say "..." --no-dedup

# Skip conflicting new facts instead of updating the stored one
neatmem demo --say "..." --say "..." --dedup-resolver skip

# Lower the similarity threshold for candidate recall (default 0.40)
neatmem demo --say "..." --say "..." --dedup-recall-threshold 0.4
```

See the [Configuration](configuration.md) table for the full list of serve
flags and their environment-variable equivalents.

## Saving a run record

By default the command only prints to stdout. To keep a markdown record of a
run (inputs, configuration, per-rep decisions, final store):

```bash
neatmem demo my-case.json --reps 5 --output records/my-case.md
```

The file is written to exactly the path given.

All options:

```bash
neatmem demo --help
```
