import os

import uvicorn

from app import MODEL_ID, MODEL_REVISION, app, _load_model


def _env_flag(name: str, default: str = "1") -> bool:
    v = os.getenv(name, default).strip().lower()
    return v not in ("0", "false", "no", "off")

def _apply_proxy_env() -> None:
    http_proxy = (os.getenv("HTTP_PROXY") or "").strip()
    https_proxy = (os.getenv("HTTPS_PROXY") or "").strip()
    no_proxy = (os.getenv("NO_PROXY") or "").strip()

    if http_proxy:
        if not https_proxy:
            os.environ["HTTPS_PROXY"] = http_proxy
        if not os.getenv("http_proxy"):
            os.environ["http_proxy"] = http_proxy
        if not os.getenv("https_proxy"):
            os.environ["https_proxy"] = os.environ.get("HTTPS_PROXY", http_proxy)

    if not no_proxy:
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"
    if not os.getenv("no_proxy"):
        os.environ["no_proxy"] = os.environ["NO_PROXY"]


def main() -> None:
    preload_model = _env_flag("PRELOAD_MODEL", "1")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    _apply_proxy_env()

    if preload_model:
        print(f"[startup] Preloading model: {MODEL_ID} (revision={MODEL_REVISION!r})")
        _load_model(MODEL_ID, MODEL_REVISION)
        print("[startup] Model ready; starting HTTP server...")
    else:
        print("[startup] PRELOAD_MODEL=0; starting HTTP server without preloading...")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
