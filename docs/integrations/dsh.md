# DeepSeek Harness Integration

NeatMem includes a DeepSeek Harness (dsh) plugin under `dsh/`. With the NeatMem server running at `http://localhost:8790`:

```bash
dsh plugin --profile web add @neatmem/dsh-neatmem
```

Restart dsh to load the plugin — memory is on. (Using the headless CLI or another profile instead of the web UI? Swap `web` for that profile's name.) Verify with `dsh --profile web --dump-config` (a `neatmem-dsh` row appears).

The plugin is pure TypeScript — no native dependencies and no build approvals. It works with zero configuration (`baseUrl=http://localhost:8790`, `userId=default`); override per profile in `~/.dsh/profiles/<profile>/cordis.patch.yml`:

```yaml
- id: neatmem-dsh
  config:
    userId: myname
```

Each direct-user turn gets one bounded automatic recall (fail-open, injected as a source-labelled message), every finished turn is forwarded to the server's `/v1/messages/` batching pipeline, and the agent gets five memory tools (`memory_search`, `memory_list`, `memory_get`, `memory_update`, `memory_delete`). Verified against dsh `0.1.5-rc.2`.

See [dsh/README.md](https://github.com/kanhaoning/NeatMem/blob/main/dsh/README.md) for the full configuration reference and development setup.
