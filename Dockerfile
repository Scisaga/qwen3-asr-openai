# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /app/requirements.txt

COPY app.py /app/app.py
COPY mcp_server.py /app/mcp_server.py
COPY transcription_service.py /app/transcription_service.py
COPY text_normalize.py /app/text_normalize.py
COPY server.py /app/server.py
COPY static /app/static
EXPOSE 12301
CMD ["python", "-u", "server.py"]
