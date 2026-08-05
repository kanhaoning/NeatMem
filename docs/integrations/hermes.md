# Hermes Integration

NeatMem includes a Hermes Agent memory provider under `hermes/`. With the NeatMem server running at `http://localhost:8790`:

```bash
hermes plugins install kanhaoning/NeatMem/hermes --enable
hermes config set memory.provider neatmem
```

The plugin registers five memory tools (`neatmem_search`, `neatmem_add`, `neatmem_list`, `neatmem_update`, `neatmem_delete`) and recalls memories automatically on each turn. Optional configuration via `~/.hermes/neatmem.json`:

```json
{
  "base_url": "http://localhost:8790",
  "user_id": "myname",
  "rerank": true
}
```

Verify: tell Hermes "remember that I prefer dark themes", then ask about it in a new session. See [hermes/README.md](https://github.com/kanhaoning/NeatMem/blob/main/hermes/README.md) for the full configuration reference and troubleshooting.
