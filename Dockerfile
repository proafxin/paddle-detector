FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
RUN uv sync --no-install-project --no-cache

COPY server.py /app/server.py

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8103

CMD ["fastapi", "run", "server.py", "--host", "0.0.0.0", "--port", "8103"]
