import os

import uvicorn

from app import MODEL_ID, MODEL_REVISION, app, _load_model


def _env_flag(name: str, default: str = "1") -> bool:
    v = os.getenv(name, default).strip().lower()
    return v not in ("0", "false", "no", "off")


def main() -> None:
    preload_model = _env_flag("PRELOAD_MODEL", "1")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    if preload_model:
        print(f"[startup] Preloading model: {MODEL_ID} (revision={MODEL_REVISION!r})")
        _load_model(MODEL_ID, MODEL_REVISION)
        print("[startup] Model ready; starting HTTP server...")
    else:
        print("[startup] PRELOAD_MODEL=0; starting HTTP server without preloading...")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

