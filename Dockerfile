# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
ARG PIP_INDEX_URL=https://pypi.org/simple

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    for i in 1 2 3; do \
      pip install --index-url "${PIP_INDEX_URL}" --retries 10 --timeout 120 -r /app/requirements.txt && exit 0; \
      echo "pip install failed (${i}/3), retrying..."; \
      sleep $((i * 5)); \
    done; \
    exit 1

COPY app.py /app/app.py
COPY mcp_server.py /app/mcp_server.py
COPY transcription_service.py /app/transcription_service.py
COPY text_normalize.py /app/text_normalize.py
COPY server.py /app/server.py
COPY static /app/static
EXPOSE 12301
CMD ["python", "-u", "server.py"]
