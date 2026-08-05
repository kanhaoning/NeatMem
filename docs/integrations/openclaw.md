# OpenClaw Integration

NeatMem includes an OpenClaw plugin under `openclaw/`. This requires a git clone of the repository (the plugin source is not included in the pip package). Build it and install it as a linked local plugin during development:

```bash
cd /path/to/NeatMem/openclaw
npm install
npm run build

cd /path/to/NeatMem
openclaw plugins install ./openclaw --link
```

After changing plugin TypeScript source, rebuild before reinstalling or restarting OpenClaw.

The plugin id is `openclaw-neatmem`. It talks to NeatMem through the local mem0-compatible HTTP API.

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
          "mode": "platform",
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
openclaw mem0 status
```

The CLI command remains `openclaw mem0` for compatibility, but the active plugin id should be `openclaw-neatmem` and the backend should point to `http://localhost:8790`.
