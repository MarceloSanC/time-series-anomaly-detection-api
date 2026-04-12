PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
UVICORN ?= $(PYTHON) -m uvicorn

.PHONY: install test coverage run lint docker-build docker-up docker-test-build docker-test

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTEST) -v

coverage:
	$(PYTEST) --cov=app --cov-report=html --cov-report=term-missing
	@echo "Report: htmlcov/index.html"

run:
	$(UVICORN) app.main:app --host 0.0.0.0 --port 8000 --reload

lint:
	$(PYTHON) -m compileall app tests

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-test-build:
	docker compose --profile test build api-tests

docker-test:
	docker compose --profile test run --rm api-tests
