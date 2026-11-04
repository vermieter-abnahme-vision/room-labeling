.PHONY: evaluate dev-up dev-down dev-logs dev-demo dev-ingest dev-embed dev-train dev-export install dev migrate ingest embed train zeroshot import-library export materialize test lint typecheck audit check

# ML steps need the optional extras; `uv run --extra` installs them on demand.
ML = uv run --extra probe --extra siglip

install:
	uv sync --extra probe --extra siglip

dev:
	uv run alembic upgrade head
	uv run uvicorn app.main:run_factory --factory --reload --port 8000

migrate:
	uv run alembic upgrade head

ingest: migrate
	uv run python -m scripts.ingest

embed: migrate
	$(ML) python -m scripts.embed

train: migrate
	$(ML) python -m scripts.train_probe

zeroshot: migrate
	$(ML) python -m scripts.train_probe --zero-shot

# e.g. make import-library DATASET=places365 ROOT=./library_data ARGS="--download"
import-library: migrate
	uv run python -m scripts.import_library --dataset $(DATASET) --root $(ROOT) $(ARGS)

export: migrate
	uv run python -m scripts.export $(ARGS)

# e.g. make materialize MANIFEST=var/exports/<stamp>/manifest.jsonl
materialize: migrate
	uv run python -m scripts.materialize_imagefolder $(MANIFEST) $(ARGS)

# e.g. make evaluate MANIFEST=var/exports/<stamp>/manifest.jsonl ARGS=--zero-shot
evaluate:
	$(ML) python -m scripts.evaluate $(MANIFEST) $(ARGS)

test:
	uv run pytest --cov=app --cov-report=term-missing

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

audit:
	uv export --frozen --no-dev --all-extras --no-emit-project --no-hashes -o /tmp/rl-requirements.txt
	sed -i 's/+cpu//' /tmp/rl-requirements.txt
	uv run pip-audit -r /tmp/rl-requirements.txt --no-deps --disable-pip

check: lint typecheck test

# --- Local end-to-end stack (docker-compose.dev.yml: app + Dex OIDC) -------------------------
DEV = docker compose -f docker-compose.dev.yml

dev-up:
	$(DEV) up -d --build
	@echo "App: http://localhost:8000  (login: alice@example.test / password, admin@example.test is admin)"

dev-down:
	$(DEV) down -v

dev-logs:
	$(DEV) logs -f

dev-demo:
	uv run python -m scripts.demo_data

dev-ingest:
	$(DEV) run --rm app python -m scripts.ingest

dev-embed:
	$(DEV) run --rm app python -m scripts.embed

dev-train:
	$(DEV) run --rm app python -m scripts.train_probe

dev-export:
	$(DEV) run --rm app python -m scripts.export
