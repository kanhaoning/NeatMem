# API Reference

## Health check

```bash
curl http://localhost:8790/health
```

## Add memory

```bash
curl -X POST http://localhost:8790/v1/memories/ \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "My name is Alex and I work on agent memory systems."},
      {"role": "assistant", "content": "Nice to meet you, Alex."}
    ],
    "user_id": "default_user",
    "infer": true
  }'
```

## Search memory

```bash
curl -X POST http://localhost:8790/v2/memories/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Alex working on?",
    "filters": {"user_id": "default_user"},
    "top_k": 10,
    "threshold": 0.1
  }'
```

## List memories

```bash
curl -X POST http://localhost:8790/v2/memories/ \
  -H "Content-Type: application/json" \
  -d '{
    "filters": {"user_id": "default_user"},
    "page": 1,
    "page_size": 100
  }'
```

## Get memory

```bash
curl http://localhost:8790/v1/memories/{memory_id}/
```

## Update memory

```bash
curl -X PUT http://localhost:8790/v1/memories/{memory_id}/ \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Alex works on local-first agent memory systems.",
    "metadata": {"source": "manual_update"}
  }'
```

## Delete memory

```bash
curl -X DELETE http://localhost:8790/v1/memories/{memory_id}/
```
