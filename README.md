# time-series-anomaly-detection-api

API de deteccao de anomalias em series temporais com FastAPI.

## Benchmark

Para executar o benchmark de inferencia paralela (100 requests):

```bash
.venv/bin/python scripts/benchmark.py
```

O resultado e salvo em `scripts/benchmark_results.json` e inclui:
- `p50`, `p95`, `p99`, `avg`, `min`, `max` (latencia em ms)
- `throughput_rps`
- `total_duration_seconds`

Interpretacao recomendada dos percentis:
- `min` representa as primeiras requests que pegaram o servidor mais livre.
- `p99` representa requests que aguardaram mais tempo na fila sob alta concorrencia.

Com 100 requests simultaneas em servidor local single-process, essa diferenca entre `min` e `p99` e esperada e indica comportamento real sob carga.

## Documentacao

- [Docs Index](docs/README.md)
- [Roadmap](docs/project/ROADMAP.md)
- [Architecture](docs/project/ARCHITECTURE.md)
- [API Responses](docs/project/API_RESPONSES.md)
- [AI Usage](docs/ai/LLM_USAGE.md)
- [Git Protocol](docs/process/GIT_PROTOCOL.md)
