# syntax=docker/dockerfile:1

# ---- build: resolve the locked dependency set (hashes verified by uv) ---------------------
FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
COPY pyproject.toml uv.lock ./
# Default: web app + probe + SigLIP (CPU torch). Slim web-only image: --build-arg EXTRAS=""
ARG EXTRAS="--extra probe --extra siglip"
RUN uv sync --frozen --no-dev ${EXTRAS}

# ---- runtime ------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg tini \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /home/app --create-home app \
 && mkdir -p /data/images /data/db /data/embed_cache /data/models /data/exports /data/hf /inputs/images_in /inputs/videos_in \
 && chown -R app:app /data /inputs

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app app ./app
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app alembic ./alembic
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app docker/entrypoint.sh /entrypoint.sh
RUN chmod 0755 /entrypoint.sh

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=prod \
    DATABASE_URL=sqlite:////data/db/labeling.db \
    IMAGES_DIR=/data/images \
    EMBED_CACHE_DIR=/data/embed_cache \
    MODELS_DIR=/data/models \
    EXPORT_DIR=/data/exports \
    IMAGES_IN_DIR=/inputs/images_in \
    VIDEOS_IN_DIR=/inputs/videos_in \
    HF_HOME=/data/hf

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"
ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
CMD ["uvicorn", "app.main:run_factory", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
