import asyncio
import os

import uvicorn

from app import app
from transcription_service import (
    MODEL_ID,
    MODEL_REVISION,
    _apply_proxy_env,
    load_model,
    maybe_preload_backend,
    should_manage_backend_process,
)


def _env_flag(name: str, default: str = "1") -> bool:
    v = os.getenv(name, default).strip().lower()
    return v not in ("0", "false", "no", "off")

def main() -> None:
    preload_model = _env_flag("PRELOAD_MODEL", "1")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "12301"))

    _apply_proxy_env()

    if preload_model:
        if should_manage_backend_process():
            print("[startup] Preloading ASR backend replicas...")
            asyncio.run(maybe_preload_backend())
            print("[startup] Backend preload requested; starting HTTP server...")
        else:
            print(f"[startup] Preloading model: {MODEL_ID} (revision={MODEL_REVISION!r})")
            load_model(MODEL_ID, MODEL_REVISION)
            print("[startup] Model ready; starting HTTP server...")
    else:
        print("[startup] PRELOAD_MODEL=0; starting HTTP server without preloading...")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
