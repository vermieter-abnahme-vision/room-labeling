.PHONY: install dev migrate ingest embed train zeroshot import-library export materialize test lint typecheck audit check

install:
	uv sync --extra probe --extra siglip

dev:
	uv run alembic upgrade head
	uv run uvicorn app.main:run_factory --factory --reload --port 8000

migrate:
	uv run alembic upgrade head

ingest:
	uv run python -m scripts.ingest

embed:
	uv run python -m scripts.embed

train:
	uv run python -m scripts.train_probe

zeroshot:
	uv run python -m scripts.train_probe --zero-shot

# e.g. make import-library DATASET=places365 ROOT=./library_data ARGS="--download"
import-library:
	uv run python -m scripts.import_library --dataset $(DATASET) --root $(ROOT) $(ARGS)

export:
	uv run python -m scripts.export $(ARGS)

# e.g. make materialize MANIFEST=var/exports/<stamp>/manifest.jsonl
materialize:
	uv run python -m scripts.materialize_imagefolder $(MANIFEST) $(ARGS)

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
