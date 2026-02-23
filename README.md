# Qwen3-ASR：OpenAI 兼容接口 + Web 上传界面（极简）

## 功能
- OpenAI 兼容接口：`POST /v1/audio/transcriptions`
- 内置文件上传 Web UI：`GET /`
- 通过 HuggingFace 缓存自动下载模型（将 `./models` 挂载到 `/models`）
- 支持音频/视频输入；使用 ffmpeg 抽取音频并转换为 16k 单声道 wav
- 可选模型热重载：`POST /admin/reload`（由 `ADMIN_TOKEN` 保护）

## 快速开始
```bash
docker compose up -d --build
```

说明：容器启动时会自动下载并加载模型（首次启动可能需要较长时间）；模型就绪后才会开始对外提供 HTTP 服务。

如果机器需要走代理才能访问 HuggingFace，可在同目录创建 `.env`（或启动前导出环境变量）：
```bash
HTTP_PROXY=http://127.0.0.1:7890
# 可选：不走代理的地址（默认：localhost,127.0.0.1）
# NO_PROXY=localhost,127.0.0.1
```

打开：
- Web UI：http://localhost:8000/
- 健康检查：http://localhost:8000/health

## 切换模型（需重启）
在 `docker-compose.yml` 中修改 `MODEL_ID`，然后：
```bash
docker compose up -d
```

## 模型热重载（无需重启）
```bash
curl -X POST http://localhost:8000/admin/reload \
  -H "Content-Type: application/json" \
  -H "x-admin-token: change-me" \
  -d '{"model_id":"Qwen/Qwen3-ASR-0.6B"}'
```

## 多 GPU 说明
- `deploy.resources.reservations.devices.count` 控制容器内可见的 GPU 数量。
- `DEVICE_MAP` 控制模型放到哪一张（或哪几张）可见的 GPU 上：
  - `cuda:0` -> 使用容器内第 1 张可见 GPU
  - `cuda:1` -> 使用容器内第 2 张可见 GPU
  - `auto` -> 交给底层 HF accelerate 决定（可能会根据模型/权重在可见 GPU 间切分）

如果宿主机有两张 GPU，但你只想用第 2 张，可以设置 `count: 1`，并在环境变量里设置：
- `NVIDIA_VISIBLE_DEVICES: "1"`，同时保持 `DEVICE_MAP=cuda:0`（因为容器里只“看到”那一张 GPU）。
