# CUDA comes from the paddlepaddle-gpu / nvidia-* wheels, so no CUDA base image is needed:
# the host only has to provide the NVIDIA driver (run with `--gpus all`).

FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=600 \
    UV_HTTP_RETRIES=10

WORKDIR /app

COPY pyproject.toml uv.lock /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Drop what a single-GPU inference server never loads: headers and sources, multi-GPU
# communication (NCCL), and kernels/libraries that are only dlopen'd by ops the layout
# detector does not use (FlashAttention, cuSPARSELt, cuFFT, multi-GPU cuSOLVER, profiling).
RUN cd /app/.venv/lib/python3.13/site-packages \
    && rm -rf paddle/include nvidia/*/include nvidia/nccl nvidia/cusparselt \
        paddle/libs/libflashattn.so paddle/libs/libflashattnv3.so paddle/libs/libflashmaskv2.so \
        nvidia/cu13/lib/libcufft* nvidia/cu13/lib/libcusolverMg* nvidia/cu13/lib/libnvrtc.alt* \
        nvidia/cu13/lib/libnvperf* \
    && find . \( -name '*.cpp' -o -name '*.pyx' -o -name '*.pxd' -o -name '*.h' \) -delete


FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY server.py /app/server.py

ENV PATH="/app/.venv/bin:${PATH}" \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

EXPOSE 8103

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8103"]
