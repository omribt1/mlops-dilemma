# Production image for `dilemma ui` (Cloud Run). Build: docker build -t aiops-dilemma .
# No credentials are baked in: on Cloud Run the app uses the runtime service account (metadata server); locally,
# mount an ADC file read-only and point GOOGLE_APPLICATION_CREDENTIALS at it.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Dependencies first (cached layer), exactly as pinned in uv.lock.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project

# The app. Installed editable, so the server finds blog/example (the read-only example session) next to src/.
COPY src ./src
COPY blog/example ./blog/example
COPY examples ./examples
RUN uv sync --locked --no-dev \
 && useradd --create-home --uid 10001 --user-group app \
 && mkdir -p /data && chown app:app /data

USER app
ENV PATH=/app/.venv/bin:$PATH \
    HOME=/home/app \
    DILEMMA_RUNS_DIR=/data/runs \
    PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "exec dilemma ui --host 0.0.0.0 --port ${PORT:-8080}"]
