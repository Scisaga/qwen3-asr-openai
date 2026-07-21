import os

import uvicorn

from app import app
from transcription_service import _apply_proxy_env


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "12301"))

    _apply_proxy_env()
    print("[startup] Starting HTTP server; ASR preload runs in the app background task when enabled.")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
