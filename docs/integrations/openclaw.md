# OpenClaw Integration

With the NeatMem server running at `http://localhost:8790`:

```bash
openclaw plugins install @neatmem/openclaw-neatmem
openclaw neatmem init
```

Then restart the gateway (`openclaw gateway restart`) to load the plugin.

`init` works with zero flags: it writes `apiKey=neatmem-local`, `baseUrl=http://localhost:8790`, and your OS username as `userId`, then validates against the server. Override with `--api-key`, `--user-id`, or `--base-url`.

Example OpenClaw configuration:

```json
{
  "plugins": {
    "slots": {
      "memory": "openclaw-neatmem"
    },
    "entries": {
      "openclaw-neatmem": {
        "enabled": true,
        "config": {
          "apiKey": "neatmem-local",
          "userId": "default_user",
          "baseUrl": "http://localhost:8790"
        }
      }
    }
  }
}
```

Then check:

```bash
openclaw neatmem status
```

The plugin id is `openclaw-neatmem`. It talks to NeatMem through the local mem0-compatible HTTP API. For full CLI/tool reference and building from source, see the [plugin README](https://github.com/kanhaoning/NeatMem/blob/main/openclaw/README.md).
