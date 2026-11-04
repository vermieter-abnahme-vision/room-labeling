# Contributing

## Ground rules

* **Never commit data**: no photos, videos, databases, embeddings, model files, `.env`.
  Tests use synthetic images generated at runtime. `.gitignore`, the pre-commit hook and the CI
  secret scan are safety nets, not a licence to be careless.
* Business logic lives in `app/services/` (no FastAPI imports there). Routers stay thin.
* Everything is typed (`mypy --strict`) and covered by tests; services need unit tests.

## Setup

```bash
uv sync --extra probe          # runtime + dev tools (add --extra siglip for the real embedder)
uv run pre-commit install      # same checks as CI, locally
make check                     # ruff + mypy + pytest
```

## Pull requests

1. Branch from `main`, keep PRs focused.
2. `make check` must pass; add/adjust tests for behaviour changes.
3. Schema changes need an Alembic migration (`uv run alembic revision --autogenerate -m "..."`);
   `tests/test_migrations.py` fails if models and migrations diverge.
4. Dependency changes: update `pyproject.toml`, run `uv lock`, commit `uv.lock`.
5. GitHub Actions are pinned to commit SHAs (Dependabot keeps them current).

## Security issues

See [SECURITY.md](SECURITY.md) – do not file them as public issues.
