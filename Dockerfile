FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

COPY pyproject.toml README.md ./
COPY app ./app

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

RUN mkdir -p /app/storage && chown -R appuser:appgroup /app

FROM base AS test

COPY tests ./tests
RUN pip install --no-cache-dir "pytest==8.2.0" "pytest-asyncio==0.23.0" "pytest-cov==5.0.0"

USER appuser

CMD ["pytest", "-v"]

FROM base AS runtime

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host ${APP_HOST:-0.0.0.0} --port ${APP_PORT:-8000}"]
