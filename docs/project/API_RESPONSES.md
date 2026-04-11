# API Responses

Este documento resume os responses HTTP esperados pela API com base em `docs/context/openapi_spec.yaml`.

## Train Response

Endpoint: `POST /fit/{series_id}`

Response (`200`):

```json
{
  "series_id": "sensor_abc",
  "version": "v1",
  "points_used": 120
}
```

Notas:
- `points_used` e o nome oficial no contrato HTTP.
- Internamente o service usa `n_samples` e a camada API faz o mapeamento para `points_used`.
- `training_duration_ms` e medido internamente no service e representa apenas o tempo efetivo de treino/persistencia, sem tempo de espera por lock.

## Predict Response

Endpoint: `POST /predict/{series_id}`

Response (`200`):

```json
{
  "anomaly": false,
  "model_version": "v1"
}
```

Notas:
- `model_version` e o nome oficial no contrato HTTP.
- Internamente o service usa `version` e a camada API faz o mapeamento.

## Healthcheck Response

Endpoint: `GET /healthcheck`

Response (`200`):

```json
{
  "series_trained": 2,
  "inference_latency_ms": {
    "avg": 7.2,
    "p95": 14.1
  },
  "training_latency_ms": {
    "avg": 23.7,
    "p95": 40.5
  }
}
```

Notas:
- O contrato define os blocos de latencia para inferencia e treino.
- Campos operacionais internos adicionais podem existir internamente, mas o response HTTP deve seguir o OpenAPI.
