---
name: pause
description: Pause NeatMem memory capture on this machine. Use when the user wants to stop memories being recorded, for example for private work or experiments.
disable-model-invocation: true
---

# Pause memory capture

To pause (hooks stop capturing and sending session content):

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/core/memory_cli.py" --harness "claude-code" --plugin-data-dir "${CLAUDE_PLUGIN_DATA}" pause
```

Confirm the new state back to the user, and remind them that already-created
memories still exist and remain searchable. Pending unsent packets are held
while paused, not expired, and are delivered after capture is turned back on.
To turn capture back on, use `/neatmem:unpause`.
