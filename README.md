# paddle-detector

A FastAPI service that runs the PaddleOCR layout detector (`PP-DocLayoutV3`) on a page image and returns the regions it finds with their bounding boxes.

## Run

```bash
uv sync --all-groups --all-packages --all-extras
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True uv run uvicorn server:app --port 8103
```

The first start downloads the model weights to `~/.paddlex`. The service needs an NVIDIA GPU with CUDA 13 support (`paddlepaddle-gpu` comes from the cu130 index).

With Docker:

```bash
docker build -t paddle-detector .
docker run --gpus all -p 8103:8103 -v paddlex-models:/root/.paddlex paddle-detector
```

## API

`GET /health` returns `{"status": "ok"}` once the model is loaded.

`POST /detect` takes the raw image bytes (PNG, JPEG or any format OpenCV can decode) as the request body:

```bash
curl -X POST --data-binary @page.png localhost:8103/detect
```

```json
[
  {"label": "paragraph_title", "score": 0.88, "bbox": [0.086, 0.072, 0.602, 0.119], "order": 1},
  {"label": "text", "score": 0.72, "bbox": [0.087, 0.181, 0.936, 0.56], "order": 2}
]
```

`bbox` is `[x1, y1, x2, y2]`, normalized to 0–1 by the image width and height. `order` is the reading order, or `null` for regions that have none. An empty or undecodable body returns `400`.

## Updating dependencies

`./update_deps.sh` upgrades the lockfile and raises the lower bounds in `pyproject.toml` to match.
