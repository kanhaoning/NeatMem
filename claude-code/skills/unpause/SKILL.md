---
name: unpause
description: Resume NeatMem memory capture after it was paused with /neatmem:pause.
disable-model-invocation: true
---

# Unpause memory capture

Resume memory capture for this machine.

Run:

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/core/memory_cli.py" --harness "claude-code" --plugin-data-dir "${CLAUDE_PLUGIN_DATA}" resume
```

Confirm to the user that capture is active again. New sessions record evidence and
create memories as normal; nothing that happened while paused is retroactively
captured.
