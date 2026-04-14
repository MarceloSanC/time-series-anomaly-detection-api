# API Responses

<<<<<<< ours
Este documento resume os responses HTTP esperados pela API com base em `docs/context/openapi_spec.yaml`.

## Train Response

Endpoint: `POST /fit/{series_id}`

Response (`200`):
=======
Este documento resume os responses HTTP esperados da API.

Observacao de contrato:
- Rotas base do challenge (`/fit`, `/predict`, `/healthcheck`) seguem `docs/context/openapi_spec.yaml`.
- Rotas `/models*` e `/plot` sao extensoes aditivas do projeto.

## Train (`POST /fit/{series_id}`)

### Exemplos de path

Com parametros obrigatorios:

```bash
/fit/sensor_XYZ
```

### Exemplo de request body

```json
{
  "timestamps": [1700000001, 1700000002, 1700000003],
  "values": [10.0, 10.2, 10.1]
}
```

### Response `200`
>>>>>>> theirs

```json
{
  "series_id": "sensor_abc",
  "version": "v1",
  "points_used": 120
}
```

<<<<<<< ours
Notas:
- `points_used` e o nome oficial no contrato HTTP.
- Internamente o service usa `n_samples` e a camada API faz o mapeamento para `points_used`.
- `training_duration_ms` e medido internamente no service e representa apenas o tempo efetivo de treino/persistencia, sem tempo de espera por lock.

## Predict Response

Endpoint: `POST /predict/{series_id}`

Response (`200`):
=======
### Erros de Data Quality (escopo de treino)

Retornados como `400` com payload padronizado de erro:
- `FLAT_LINE_DETECTED`: janela final da serie contem valores identicos (possivel sensor desconectado). Ativo apenas quando `FLAT_LINE_ENABLED=true`.
- `TEMPORAL_GAP_DETECTED`: intervalo maximo entre timestamps excede `MAX_TEMPORAL_GAP_FACTOR x mediana(intervalos)`. Ativo apenas quando `TEMPORAL_GAP_ENABLED=true`.

## Predict (`POST /predict/{series_id}`)

### Exemplos de path

Com parametros obrigatorios:

```bash
/predict/sensor_XYZ
```

Com parametro opcional:

- `version` (quando omitido, usa a versao mais recente da serie).

```bash
/predict/sensor_XYZ?version=v1
```

### Exemplo de request body

```json
{
  "timestamp": 1700000100,
  "value": 99.0
}
```

### Response `200`
>>>>>>> theirs

```json
{
  "anomaly": false,
  "model_version": "v1"
}
```

<<<<<<< ours
Notas:
- `model_version` e o nome oficial no contrato HTTP.
- Internamente o service usa `version` e a camada API faz o mapeamento.

## Healthcheck Response

Endpoint: `GET /healthcheck`

Response (`200`):
=======

## Healthcheck (`GET /healthcheck`)

### Exemplos de path

Path do endpoint:

```bash
/healthcheck
```

### Response `200`
>>>>>>> theirs

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

<<<<<<< ours
Notas:
- O contrato define os blocos de latencia para inferencia e treino.
- Campos operacionais internos adicionais podem existir internamente, mas o response HTTP deve seguir o OpenAPI.

## Validation Error Codes

Os codigos abaixo sao retornados exclusivamente por `POST /fit/{series_id}` no payload padronizado de erro de validacao (`400`):

- `FLAT_LINE_DETECTED`: janela final da serie contem valores identicos (possivel sensor desconectado). Ativo apenas quando `FLAT_LINE_ENABLED=true`.
- `TEMPORAL_GAP_DETECTED`: intervalo maximo entre timestamps excede o fator configurado sobre o intervalo mediano. Ativo apenas quando `TEMPORAL_GAP_ENABLED=true`.
=======
## Plot (`GET /plot`)

### Exemplos de path

Com parametro obrigatorio:

```bash
/plot?series_id=sensor_XYZ
```

Com parametro opcional:

- `version` (quando omitido, usa a versao mais recente).

```bash
/plot?series_id=sensor_XYZ&version=v1
```

### Response `200`

- `Content-Type: image/png` (arquivo PNG em streaming com pontos de treino, media e limites +-3 sigma).

### Erro relevante

- `422 PLOT_DATA_UNAVAILABLE`: metadata indisponivel/incompleta para gerar grafico.

## Models (`GET /models`)

### Exemplos de path

Path base:

```bash
/models
```

Com parametro opcional:

- `strict` (`false` por padrao).
  - `strict=false`: ignora series com metadata incompleta e retorna as validas.
  - `strict=true`: fail-fast quando encontrar metadata incompleta.

```bash
/models?strict=true
```

### Response `200`

```json
[
  {
    "series_id": "sensor_A",
    "latest_version": "v2",
    "n_samples": 120,
    "trained_at": "2026-04-14T12:00:00Z"
  }
]
```

### Erro relevante

- `422 INCOMPLETE_MODEL_METADATA` (quando `strict=true` e houver serie inconsistente).

## Model Detail (`GET /models/{series_id}`)

### Exemplos de path

Com parametro obrigatorio:

```bash
/models/sensor_XYZ
```

### Response `200`

```json
{
  "series_id": "sensor_A",
  "latest_version": "v2",
  "versions": ["v1", "v2"],
  "trained_at": "2026-04-14T12:00:00Z",
  "n_samples": 120,
  "data_quality": {
    "n_samples": 120,
    "mean": 15.2,
    "std": 1.8,
    "min_value": 10.0,
    "max_value": 19.1,
    "time_span_seconds": 119,
    "points_per_second": 1.0084
  }
}
```

### Erro relevante

- `404 SERIES_NOT_FOUND`.

## Model Version Metadata (`GET /models/{series_id}/versions/{version}`)

### Exemplos de path

Com parametros obrigatorios:

```bash
/models/sensor_XYZ/versions/v1
```

Com parametro opcional:

- `include_data` (`false` por padrao).
  - `include_data=false`: exclui `training_data`.
  - `include_data=true`: inclui `training_data` (ou `[]` para metadata legada sem esse campo).

```bash
/models/sensor_XYZ/versions/v1?include_data=true
```

### Response `200`

```json
{
  "version": "v2",
  "mean": 15.2,
  "std": 1.8,
  "n_samples": 120,
  "trained_at": "2026-04-14T12:00:00Z",
  "training_duration_ms": 24.6,
  "data_range": {
    "min_timestamp": 1700000001,
    "max_timestamp": 1700000120
  }
}
```

### Erro relevante

- `404 VERSION_NOT_FOUND`.
>>>>>>> theirs
