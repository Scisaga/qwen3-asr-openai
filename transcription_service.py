import asyncio
import base64
import binascii
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from text_normalize import normalize_zh_numbers

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
MODEL_REVISION = os.getenv("MODEL_REVISION")
DEVICE_MAP = os.getenv("DEVICE_MAP", "cuda:0")
DTYPE = os.getenv("DTYPE", "bfloat16")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "1"))
MAX_CONCURRENT_TRANSCRIBE = int(os.getenv("MAX_CONCURRENT_TRANSCRIBE", "1"))
MAX_BACKEND_IN_FLIGHT_PER_REPLICA = max(1, int(os.getenv("MAX_BACKEND_IN_FLIGHT_PER_REPLICA", "2")))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
CHUNK_SECONDS = float(os.getenv("CHUNK_SECONDS", "600"))
CHUNK_OVERLAP_SECONDS = float(os.getenv("CHUNK_OVERLAP_SECONDS", "1"))
CONTEXT_TAIL_CHARS = int(os.getenv("CONTEXT_TAIL_CHARS", "200"))
NORMALIZE_ZH_NUMBERS = (
    os.getenv("NORMALIZE_ZH_NUMBERS", "1").strip().lower()
    not in ("0", "false", "no", "off")
)
MCP_MAX_INPUT_BYTES = int(os.getenv("MCP_MAX_INPUT_BYTES", str(32 * 1024 * 1024)))
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT") or os.getenv("ASR_BACKEND_PORT", "8001"))
BACKEND_START_TIMEOUT = int(os.getenv("BACKEND_START_TIMEOUT", "600"))
BACKEND_HTTP_TIMEOUT = float(os.getenv("BACKEND_HTTP_TIMEOUT", "3600"))
BACKEND_POLL_INTERVAL = float(os.getenv("BACKEND_POLL_INTERVAL", "1.0"))
AUTO_BACKEND_REPLICAS_ENV = "AUTO_BACKEND_REPLICAS"
BACKEND_REPLICA_COUNT_ENV = "BACKEND_REPLICA_COUNT"
ASR_BACKEND_WORKER_ENV = "ASR_BACKEND_WORKER"
MANAGE_BACKEND_PROCESS_ENV = "MANAGE_BACKEND_PROCESS"

_model = None
_current_model_id = MODEL_ID
_current_model_revision = MODEL_REVISION
_transcribe_sem = asyncio.Semaphore(max(1, MAX_CONCURRENT_TRANSCRIBE))
_backend_lock = threading.RLock()
_backend_replicas: list["BackendReplica"] = []
_backend_router_index = 0
_backend_started_at: Optional[float] = None
_backend_ready = False
_backend_last_error = ""

_SAFE_SUFFIX_RE = re.compile(r"^\.[a-z0-9]{1,15}$")
_MIME_SUFFIX_OVERRIDES = {
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/webm": ".webm",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/quicktime": ".mov",
}


class InputValidationError(ValueError):
    pass


class InputTooLargeError(InputValidationError):
    def __init__(self, size: int, max_bytes: int):
        self.size = size
        self.max_bytes = max_bytes
        super().__init__(
            "audio_base64 decoded payload is too large "
            f"({size} bytes > {max_bytes} bytes). "
            "For large audio/video files, use POST /v1/audio/transcriptions instead."
        )


class CudaOOMTranscriptionError(RuntimeError):
    def __init__(self, detail: dict):
        self.detail = detail
        super().__init__(detail.get("message", "CUDA out of memory"))


class BackendUnavailableError(RuntimeError):
    pass


class BackendTranscriptionError(RuntimeError):
    pass


@dataclass
class BackendReplica:
    replica_index: int
    port: int
    base_url: str
    device_identifier: Optional[str] = None
    process: Optional[subprocess.Popen[Any]] = None
    started_at: Optional[float] = None
    ready: bool = False
    in_flight: int = 0
    request_count: int = 0
    last_error: str = ""
    health: Optional[dict[str, Any]] = None


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off")


def _is_backend_worker() -> bool:
    return _env_flag(ASR_BACKEND_WORKER_ENV, "0")


def _apply_proxy_env(target_env: Optional[dict[str, str]] = None) -> dict[str, str]:
    env = target_env if target_env is not None else os.environ
    http_proxy = (env.get("HTTP_PROXY") or os.getenv("HTTP_PROXY") or "").strip()
    https_proxy = (env.get("HTTPS_PROXY") or os.getenv("HTTPS_PROXY") or "").strip()
    no_proxy = (env.get("NO_PROXY") or os.getenv("NO_PROXY") or "").strip()

    if http_proxy:
        env["HTTP_PROXY"] = http_proxy
        env["http_proxy"] = http_proxy
        if not https_proxy:
            https_proxy = http_proxy
        env["HTTPS_PROXY"] = https_proxy
        env["https_proxy"] = https_proxy

    env["NO_PROXY"] = no_proxy or "localhost,127.0.0.1"
    env["no_proxy"] = env["NO_PROXY"]
    return env


def _is_cuda_device_map(device_map: str) -> bool:
    normalized = (device_map or "").strip().lower()
    return normalized == "cuda" or normalized.startswith("cuda:")


def _backend_replica_count_override() -> Optional[int]:
    raw_value = os.getenv(BACKEND_REPLICA_COUNT_ENV, "").strip()
    if not raw_value:
        return None
    try:
        count = int(raw_value)
    except ValueError as exc:
        raise BackendUnavailableError(
            f"Invalid {BACKEND_REPLICA_COUNT_ENV}: expected positive integer, got {raw_value!r}."
        ) from exc
    if count < 1:
        raise BackendUnavailableError(
            f"Invalid {BACKEND_REPLICA_COUNT_ENV}: expected positive integer, got {raw_value!r}."
        )
    return count


def _detect_visible_gpu_identifiers() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env=_apply_proxy_env(dict(os.environ)),
        )
        if result.returncode == 0:
            gpu_count = sum(1 for line in result.stdout.splitlines() if line.strip().startswith("GPU "))
            if gpu_count > 0:
                return [str(index) for index in range(gpu_count)]
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass

    for env_name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        raw_value = os.getenv(env_name, "").strip()
        if not raw_value:
            continue
        lowered = raw_value.lower()
        if lowered in ("none", "void"):
            return []
        if lowered == "all":
            continue
        identifiers = [token.strip() for token in raw_value.split(",") if token.strip()]
        if identifiers:
            if any(identifier.startswith("GPU-") for identifier in identifiers):
                return [str(index) for index in range(len(identifiers))]
            return identifiers
    return []


def should_manage_backend_process() -> bool:
    if _is_backend_worker():
        return False

    raw_value = os.getenv(MANAGE_BACKEND_PROCESS_ENV, "").strip().lower()
    if raw_value:
        return raw_value not in ("0", "false", "no", "off")

    replica_override = _backend_replica_count_override()
    if replica_override is not None:
        return replica_override > 1
    if not _env_flag(AUTO_BACKEND_REPLICAS_ENV, "1"):
        return False

    return len(_detect_visible_gpu_identifiers()) > 1


def _desired_backend_device_identifiers() -> list[Optional[str]]:
    if not should_manage_backend_process():
        return []

    identifiers = _detect_visible_gpu_identifiers()
    replica_override = _backend_replica_count_override()
    if replica_override is not None:
        if identifiers and replica_override > len(identifiers):
            raise BackendUnavailableError(
                f"{BACKEND_REPLICA_COUNT_ENV}={replica_override} exceeds visible GPU count ({len(identifiers)})."
            )
        if identifiers:
            return identifiers[:replica_override]
        return [str(index) for index in range(replica_override)]

    if len(identifiers) <= 1:
        if os.getenv(MANAGE_BACKEND_PROCESS_ENV, "").strip().lower() not in ("", "0", "false", "no", "off"):
            return identifiers or [None]
        return []
    return identifiers


def _build_backend_replicas_layout() -> list[BackendReplica]:
    return [
        BackendReplica(
            replica_index=index,
            port=BACKEND_PORT + index,
            base_url=f"http://{BACKEND_HOST}:{BACKEND_PORT + index}",
            device_identifier=device_identifier,
        )
        for index, device_identifier in enumerate(_desired_backend_device_identifiers())
    ]


def _same_replica_layout(current: list[BackendReplica], desired: list[BackendReplica]) -> bool:
    if len(current) != len(desired):
        return False
    for current_replica, desired_replica in zip(current, desired):
        if current_replica.port != desired_replica.port:
            return False
        if current_replica.device_identifier != desired_replica.device_identifier:
            return False
    return True


def _replica_alive(replica: BackendReplica) -> bool:
    return replica.process is not None and replica.process.poll() is None


def _refresh_backend_summary_locked() -> None:
    global _backend_ready, _backend_last_error, _backend_started_at

    ready_replicas = [replica for replica in _backend_replicas if replica.ready]
    _backend_ready = bool(ready_replicas)
    _backend_last_error = "" if ready_replicas else next(
        (replica.last_error for replica in _backend_replicas if replica.last_error),
        _backend_last_error,
    )
    started_at_values = [replica.started_at for replica in _backend_replicas if replica.started_at is not None]
    _backend_started_at = min(started_at_values) if started_at_values else None


def _ensure_backend_layout_locked() -> None:
    global _backend_replicas, _backend_router_index

    desired_layout = _build_backend_replicas_layout()
    if _same_replica_layout(_backend_replicas, desired_layout):
        return

    _backend_replicas = desired_layout
    _backend_router_index = 0
    _refresh_backend_summary_locked()


def _build_backend_env(replica: BackendReplica) -> dict[str, str]:
    env = _apply_proxy_env(dict(os.environ))
    env[ASR_BACKEND_WORKER_ENV] = "1"
    env["HOST"] = BACKEND_HOST
    env["PORT"] = str(replica.port)
    env["PRELOAD_MODEL"] = "1"
    env["HF_HOME"] = os.getenv("HF_HOME", "/models")
    if replica.device_identifier is not None:
        env["CUDA_VISIBLE_DEVICES"] = replica.device_identifier
        if _is_cuda_device_map(env.get("DEVICE_MAP", DEVICE_MAP)):
            # A worker sees exactly one GPU, so its local CUDA index is always 0.
            env["DEVICE_MAP"] = "cuda:0"
    return env


def _start_backend_process_locked() -> None:
    global _backend_last_error

    if not should_manage_backend_process():
        return

    _ensure_backend_layout_locked()
    if not _backend_replicas:
        _backend_last_error = "No ASR backend replicas are configured."
        raise BackendUnavailableError(_backend_last_error)

    server_path = os.path.join(os.path.dirname(__file__), "server.py")
    for replica in _backend_replicas:
        if _replica_alive(replica):
            continue

        command = [sys.executable, "-u", server_path]
        try:
            replica.process = subprocess.Popen(
                command,
                cwd=os.path.dirname(__file__) or None,
                env=_build_backend_env(replica),
                stdin=subprocess.DEVNULL,
                stdout=None,
                stderr=None,
                start_new_session=True,
            )
        except OSError as exc:
            replica.process = None
            replica.ready = False
            replica.last_error = f"Failed to start ASR backend replica {replica.replica_index}: {exc}"
            _backend_last_error = replica.last_error
            raise BackendUnavailableError(_backend_last_error) from exc

        replica.started_at = time.time()
        replica.ready = False
        replica.last_error = ""
        replica.health = None

    _backend_last_error = ""
    _refresh_backend_summary_locked()


def _stop_backend_process_locked() -> None:
    for replica in _backend_replicas:
        proc = replica.process
        replica.process = None
        replica.ready = False
        replica.health = None

        if proc is None or proc.poll() is not None:
            continue

        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    _refresh_backend_summary_locked()


def _probe_single_backend_health(base_url: str) -> tuple[bool, Optional[dict[str, Any]], str]:
    import httpx

    try:
        with httpx.Client(timeout=httpx.Timeout(3.0, connect=1.0)) as client:
            response = client.get(f"{base_url.rstrip('/')}/health")
    except httpx.HTTPError as exc:
        return False, None, str(exc)

    if response.status_code >= 400:
        return False, None, f"/health returned HTTP {response.status_code} on {base_url}"

    try:
        payload = response.json()
    except ValueError:
        return False, None, "Backend returned non-JSON health payload."

    if payload.get("model_loaded") is False:
        return False, payload, "ASR backend model is not loaded yet."
    return True, payload, ""


def _probe_backend_health() -> tuple[bool, Optional[dict[str, Any]], str]:
    global _backend_ready, _backend_last_error

    with _backend_lock:
        _ensure_backend_layout_locked()
        replicas = list(_backend_replicas)

    if not replicas:
        _backend_ready = False
        return False, None, "No ASR backend replicas are configured."

    any_healthy = False
    first_payload: Optional[dict[str, Any]] = None
    first_error = ""

    for replica in replicas:
        healthy, payload, message = _probe_single_backend_health(replica.base_url)
        replica.ready = healthy
        replica.health = payload if healthy else None
        if healthy:
            replica.last_error = ""
            if first_payload is None:
                first_payload = payload
            any_healthy = True
        elif message:
            replica.last_error = message
            if not first_error:
                first_error = message

    with _backend_lock:
        _refresh_backend_summary_locked()

    if any_healthy:
        _backend_ready = True
        _backend_last_error = ""
        return True, first_payload, ""

    _backend_ready = False
    if first_error:
        _backend_last_error = first_error
    return False, first_payload, first_error or "All ASR backend health checks failed."


async def wait_for_backend_ready(timeout_s: Optional[float] = None) -> None:
    global _backend_ready, _backend_last_error

    deadline = time.monotonic() + float(timeout_s or BACKEND_START_TIMEOUT)
    while time.monotonic() < deadline:
        with _backend_lock:
            replicas = list(_backend_replicas)

        alive_replicas = [replica for replica in replicas if _replica_alive(replica)]
        if not alive_replicas:
            exited_replica = next(
                (replica for replica in replicas if replica.process is not None and replica.process.poll() is not None),
                None,
            )
            if exited_replica is not None and exited_replica.process is not None:
                _backend_last_error = (
                    f"ASR backend replica {exited_replica.replica_index} exited "
                    f"with code {exited_replica.process.returncode}."
                )
            else:
                _backend_last_error = "ASR backend process is not running."
            _backend_ready = False
            raise BackendUnavailableError(_backend_last_error)

        healthy, _, message = await asyncio.to_thread(_probe_backend_health)
        if healthy:
            _backend_ready = True
            _backend_last_error = ""
            return

        if message:
            _backend_last_error = message
        await asyncio.sleep(BACKEND_POLL_INTERVAL)

    _backend_ready = False
    _backend_last_error = f"Timed out after {int(timeout_s or BACKEND_START_TIMEOUT)}s waiting for ASR backends."
    raise BackendUnavailableError(_backend_last_error)


async def ensure_backend_started(wait_ready: bool = True, timeout_s: Optional[float] = None) -> None:
    with _backend_lock:
        _start_backend_process_locked()
    if wait_ready:
        await wait_for_backend_ready(timeout_s=timeout_s)


async def maybe_preload_backend() -> None:
    if not should_manage_backend_process():
        return
    try:
        await ensure_backend_started(wait_ready=True)
    except Exception as exc:
        global _backend_ready, _backend_last_error
        _backend_ready = False
        _backend_last_error = str(exc)


async def shutdown_backend() -> None:
    with _backend_lock:
        _stop_backend_process_locked()


def _get_torch():
    import torch

    return torch


def _get_qwen_asr_model():
    from qwen_asr import Qwen3ASRModel

    return Qwen3ASRModel


def _torch_dtype(dtype_str: str):
    torch = _get_torch()
    s = (dtype_str or "").lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def load_model(model_id: str, revision: Optional[str] = None):
    global _model, _current_model_id, _current_model_revision

    qwen3_asr_model = _get_qwen_asr_model()
    kwargs = dict(
        torch_dtype=_torch_dtype(DTYPE),
        device_map=DEVICE_MAP,
        max_inference_batch_size=MAX_BATCH,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    if revision:
        kwargs["revision"] = revision
    _model = qwen3_asr_model.from_pretrained(model_id, **kwargs)
    _current_model_id = model_id
    _current_model_revision = revision


def ensure_model():
    global _model
    if _model is None:
        load_model(_current_model_id, _current_model_revision)


def map_language(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    normalized = lang.strip().lower()
    if normalized in ("zh", "zh-cn", "zh_cn", "chinese", "中文"):
        return "Chinese"
    if normalized in ("en", "english", "英文"):
        return "English"
    return lang


def _to_wav16k_mono(input_path: str) -> str:
    out_path = tempfile.mktemp(suffix=".wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        out_path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", errors="ignore")[-2000:])
    return out_path


def _ffprobe_duration_seconds(path: str) -> Optional[float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nk=1:nw=1",
        path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        return None
    try:
        value = float((p.stdout or "").strip())
        if math.isfinite(value) and value > 0:
            return value
    except Exception:
        return None
    return None


def _extract_wav_segment(input_wav: str, start_s: float, duration_s: float, out_wav: str) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{duration_s:.3f}",
        "-i",
        input_wav,
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        out_wav,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", errors="ignore")[-2000:])


def _split_wav_with_overlap(input_wav: str, chunk_s: float, overlap_s: float) -> tuple[list[str], str]:
    duration_s = _ffprobe_duration_seconds(input_wav)
    if not duration_s:
        return ([input_wav], "")

    chunk_s = float(chunk_s)
    overlap_s = float(overlap_s)
    if chunk_s <= 0 or overlap_s < 0 or chunk_s <= overlap_s:
        return ([input_wav], "")

    if duration_s <= chunk_s + 0.25:
        return ([input_wav], "")

    stride_s = chunk_s - overlap_s
    out_dir = tempfile.mkdtemp(prefix="qwen3asr_chunks_")
    chunks: list[str] = []

    start = 0.0
    idx = 0
    while start < duration_s - 0.05:
        seg_dur = min(chunk_s, max(0.0, duration_s - start))
        if seg_dur <= 0.05:
            break
        out_path = os.path.join(out_dir, f"chunk_{idx:06d}.wav")
        _extract_wav_segment(input_wav, start, seg_dur, out_path)
        chunks.append(out_path)
        idx += 1
        start += stride_s

    if not chunks:
        shutil.rmtree(out_dir, ignore_errors=True)
        return ([input_wav], "")

    return (chunks, out_dir)


def _max_overlap_suffix_prefix(a: str, b: str, max_len: int = 160) -> int:
    if not a or not b:
        return 0
    a_tail = a[-max_len:]
    b_head = b[:max_len]
    max_l = min(len(a_tail), len(b_head))
    for length in range(max_l, 10, -1):
        if a_tail[-length:] == b_head[:length]:
            return length
    return 0


def _merge_texts_with_overlap(texts: list[str]) -> str:
    merged = ""
    for text in texts:
        text = (text or "").strip()
        if not text:
            continue
        if not merged:
            merged = text
            continue
        overlap = _max_overlap_suffix_prefix(merged, text)
        if overlap > 0:
            merged += text[overlap:]
            continue

        joiner = ""
        if merged and not merged.endswith(("\n", "。", "！", "？", "!", "?", "…")) and not text.startswith(
            ("，", "。", "！", "？", ",", ".", "!", "?", "；", ";", "：", ":", "、")
        ):
            a_last = merged[-1]
            b_first = text[0]
            if (
                a_last.isascii()
                and b_first.isascii()
                and (a_last.isalnum() or a_last in ("%", "/"))
                and b_first.isalnum()
            ):
                joiner = " "
        elif merged and merged.endswith(("。", "！", "？", "!", "?", "…")) and not merged.endswith("\n"):
            joiner = "\n"
        merged += joiner + text
    return merged


def _clear_cuda_cache() -> None:
    try:
        torch = _get_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _is_cuda_oom_error(exc: Exception) -> bool:
    try:
        torch = _get_torch()
        oom_error = getattr(torch.cuda, "OutOfMemoryError", None)
        return oom_error is not None and isinstance(exc, oom_error)
    except Exception:
        return False


def _cuda_oom_detail(exc: Exception) -> dict:
    return {
        "error": "cuda_oom",
        "message": "CUDA 显存不足：当前模型/输入在推理时超出 GPU 可用显存。",
        "tips": [
            "将 MAX_BATCH 调小（建议 1）",
            "将 MAX_NEW_TOKENS 调小（如 128/256）",
            "换更小模型（如 Qwen/Qwen3-ASR-0.6B）或使用更多 GPU",
            "超长音频/视频建议先截短或分段再转写",
            "可尝试设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True（缓解显存碎片/保留块）",
        ],
        "exception": str(exc),
    }


def _transcribe_path(in_path: str, lang: Optional[str], context: str) -> tuple[str, Optional[str]]:
    wav_path = None
    chunk_dir = ""
    try:
        wav_path = _to_wav16k_mono(in_path)
        chunk_paths, chunk_dir = _split_wav_with_overlap(
            wav_path, CHUNK_SECONDS, CHUNK_OVERLAP_SECONDS
        )

        texts: list[str] = []
        langs: list[str] = []
        prev_tail = ""
        base_context = (context or "").strip()
        for chunk_path in chunk_paths:
            chunk_context = base_context
            if CONTEXT_TAIL_CHARS > 0 and prev_tail:
                chunk_context = f"{base_context}\n{prev_tail}" if base_context else prev_tail

            results = _model.transcribe(audio=chunk_path, context=chunk_context, language=lang)
            chunk_text = ""
            if results:
                seg_texts: list[str] = []
                for result in results:
                    result_text = (getattr(result, "text", "") or "").strip()
                    if result_text:
                        seg_texts.append(result_text)
                    result_lang = (getattr(result, "language", "") or "").strip()
                    if result_lang:
                        langs.append(result_lang)
                chunk_text = _merge_texts_with_overlap(seg_texts)
                if chunk_text:
                    texts.append(chunk_text)
            if CONTEXT_TAIL_CHARS > 0:
                prev_tail = chunk_text[-CONTEXT_TAIL_CHARS:] if chunk_text else ""
            _clear_cuda_cache()

        merged_text = _merge_texts_with_overlap(texts)
        if NORMALIZE_ZH_NUMBERS and merged_text:
            merged_text = normalize_zh_numbers(merged_text)
        merged_lang = lang
        if not merged_lang:
            uniq = [value for value in dict.fromkeys([value for value in langs if value])]
            merged_lang = ",".join(uniq) if uniq else None
        return (merged_text, merged_lang)
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass
        if chunk_dir:
            shutil.rmtree(chunk_dir, ignore_errors=True)


def _write_temp_input(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name


def guess_audio_suffix(filename: Optional[str] = None, mime_type: Optional[str] = None) -> str:
    if filename:
        ext = os.path.splitext(os.path.basename(filename))[1].lower()
        if _SAFE_SUFFIX_RE.match(ext):
            return ext

    if mime_type:
        normalized = mime_type.split(";", 1)[0].strip().lower()
        if normalized in _MIME_SUFFIX_OVERRIDES:
            return _MIME_SUFFIX_OVERRIDES[normalized]
        guessed = mimetypes.guess_extension(normalized, strict=False)
        if guessed and _SAFE_SUFFIX_RE.match(guessed):
            return guessed

    return ".bin"


def decode_audio_base64(audio_base64: str, max_bytes: int = MCP_MAX_INPUT_BYTES) -> bytes:
    if not audio_base64 or not audio_base64.strip():
        raise InputValidationError("audio_base64 is required")

    payload = audio_base64.strip()
    if payload.startswith("data:"):
        parts = payload.split(",", 1)
        if len(parts) != 2:
            raise InputValidationError("audio_base64 data URL is missing a payload separator")
        payload = parts[1]

    payload = "".join(payload.split())
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InputValidationError(
            "audio_base64 must be valid base64 or a data URL containing base64 data"
        ) from exc

    if not decoded:
        raise InputValidationError("audio_base64 decoded to an empty payload")
    if len(decoded) > max_bytes:
        raise InputTooLargeError(len(decoded), max_bytes)
    return decoded


def _backend_candidate_pool_locked() -> list[BackendReplica]:
    if not _backend_replicas:
        return []

    ready_replicas = [replica for replica in _backend_replicas if replica.ready]
    return ready_replicas or [replica for replica in _backend_replicas if _replica_alive(replica)] or list(
        _backend_replicas
    )


def _reserve_backend_replica_locked(excluded_replica_ids: set[int]) -> Optional[BackendReplica]:
    global _backend_router_index

    pool = _backend_candidate_pool_locked()
    candidates = [
        replica
        for replica in pool
        if id(replica) not in excluded_replica_ids
        and replica.in_flight < MAX_BACKEND_IN_FLIGHT_PER_REPLICA
    ]
    if not candidates:
        return None

    positions = {id(replica): index for index, replica in enumerate(pool)}
    selected = min(
        candidates,
        key=lambda replica: (
            replica.in_flight,
            (positions[id(replica)] - _backend_router_index) % len(pool),
        ),
    )
    _backend_router_index = (positions[id(selected)] + 1) % len(pool)
    selected.in_flight += 1
    selected.request_count += 1
    return selected


def _release_backend_replica_locked(
    replica: BackendReplica,
    *,
    ready: Optional[bool] = None,
    last_error: str = "",
) -> None:
    replica.in_flight = max(0, replica.in_flight - 1)
    if ready is not None:
        replica.ready = ready
    replica.last_error = last_error


def _mime_type_for_suffix(suffix: str) -> str:
    guessed = mimetypes.guess_type(f"input{suffix or '.bin'}")[0]
    return guessed or "application/octet-stream"


async def _post_transcription_to_backend(
    base_url: str,
    data: bytes,
    suffix: str,
    language: Optional[str],
    prompt: Optional[str],
) -> dict:
    import httpx

    form_data: dict[str, str] = {}
    if language is not None:
        form_data["language"] = language
    if prompt is not None:
        form_data["prompt"] = prompt

    file_suffix = suffix if _SAFE_SUFFIX_RE.match(suffix or "") else ".bin"
    files = {
        "file": (
            f"audio{file_suffix}",
            data,
            _mime_type_for_suffix(file_suffix),
        )
    }
    timeout = httpx.Timeout(BACKEND_HTTP_TIMEOUT, connect=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/v1/audio/transcriptions",
                data=form_data,
                files=files,
            )
    except httpx.HTTPError as exc:
        raise BackendUnavailableError(f"Failed to reach ASR backend {base_url}: {exc}") from exc

    if response.status_code >= 400:
        try:
            payload = response.json()
        except ValueError:
            payload = {"detail": response.text or f"Backend returned HTTP {response.status_code}"}
        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        if response.status_code == 507 and isinstance(detail, dict):
            raise CudaOOMTranscriptionError(detail)
        raise BackendTranscriptionError(
            detail if isinstance(detail, str) else f"Backend returned HTTP {response.status_code}: {detail}"
        )

    try:
        return response.json()
    except ValueError as exc:
        raise BackendTranscriptionError("ASR backend returned non-JSON response.") from exc


async def _transcribe_input_bytes_via_backend(
    data: bytes,
    suffix: str = ".bin",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> dict:
    await ensure_backend_started(wait_ready=True)

    attempted_replica_ids: set[int] = set()
    last_unavailable: Optional[BackendUnavailableError] = None
    while True:
        with _backend_lock:
            replica = _reserve_backend_replica_locked(attempted_replica_ids)

        if replica is None:
            break

        attempted_replica_ids.add(id(replica))
        print(
            "[router] dispatch "
            f"replica={replica.replica_index} url={replica.base_url} "
            f"in_flight={replica.in_flight}",
            flush=True,
        )
        try:
            result = await _post_transcription_to_backend(
                replica.base_url,
                data=data,
                suffix=suffix,
                language=language,
                prompt=prompt,
            )
        except BackendUnavailableError as exc:
            with _backend_lock:
                _release_backend_replica_locked(replica, ready=False, last_error=str(exc))
            last_unavailable = exc
            continue
        except BaseException:
            with _backend_lock:
                _release_backend_replica_locked(replica)
            raise

        with _backend_lock:
            _release_backend_replica_locked(replica, ready=True)
        return result

    if last_unavailable is not None:
        raise last_unavailable
    raise BackendUnavailableError(
        "All ASR backend replicas are busy "
        f"(max in-flight per replica: {MAX_BACKEND_IN_FLIGHT_PER_REPLICA})."
    )


async def _transcribe_input_bytes_local(
    data: bytes,
    suffix: str = ".bin",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> dict:
    ensure_model()

    input_path = _write_temp_input(data, suffix)
    mapped_language = map_language(language)
    context = (prompt or "").strip()

    try:
        try:
            async with _transcribe_sem:
                transcribe_task = asyncio.create_task(
                    asyncio.to_thread(_transcribe_path, input_path, mapped_language, context)
                )
                try:
                    merged_text, merged_lang = await asyncio.shield(transcribe_task)
                except asyncio.CancelledError:
                    try:
                        await transcribe_task
                    finally:
                        raise
        except Exception as exc:
            if _is_cuda_oom_error(exc):
                _clear_cuda_cache()
                raise CudaOOMTranscriptionError(_cuda_oom_detail(exc)) from exc
            raise BackendTranscriptionError(str(exc)) from exc

        return {
            "text": merged_text,
            "language": merged_lang,
        }
    finally:
        if input_path and os.path.exists(input_path):
            try:
                os.remove(input_path)
            except Exception:
                pass


async def transcribe_input_bytes(
    data: bytes,
    suffix: str = ".bin",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> dict:
    if should_manage_backend_process():
        return await _transcribe_input_bytes_via_backend(
            data,
            suffix=suffix,
            language=language,
            prompt=prompt,
        )
    return await _transcribe_input_bytes_local(
        data,
        suffix=suffix,
        language=language,
        prompt=prompt,
    )


async def reload_model_backend(model_id: str, revision: Optional[str] = None) -> dict:
    global _current_model_id, _current_model_revision

    if should_manage_backend_process():
        with _backend_lock:
            _stop_backend_process_locked()
            _current_model_id = model_id
            _current_model_revision = revision
            os.environ["MODEL_ID"] = model_id
            if revision:
                os.environ["MODEL_REVISION"] = revision
            else:
                os.environ.pop("MODEL_REVISION", None)
            _start_backend_process_locked()

        await wait_for_backend_ready()
        return get_health_payload()

    load_model(model_id, revision)
    return get_health_payload()


def _backend_state(healthy: bool, process_alive: bool, exit_code: Optional[int]) -> str:
    if healthy:
        return "ready"
    if process_alive:
        return "starting"
    if exit_code is not None:
        return "exited"
    return "stopped"


def _backend_message(
    healthy: bool,
    process_alive: bool,
    exit_code: Optional[int],
    probe_message: str,
) -> str:
    if healthy:
        return ""
    if process_alive:
        return (
            "ASR backend worker 已启动，但健康检查尚未就绪。"
            "这通常表示模型仍在加载权重到 GPU。"
        )
    if exit_code is not None:
        return _backend_last_error or f"ASR backend exited with code {exit_code}."
    return _backend_last_error or probe_message or "ASR backend is not reachable."


def _managed_backend_runtime_locked() -> tuple[Optional[int], Optional[int], bool, int, list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    for replica in _backend_replicas:
        process_alive = _replica_alive(replica)
        exit_code = replica.process.poll() if replica.process is not None else None
        details.append(
            {
                "replica_index": replica.replica_index,
                "base_url": replica.base_url,
                "port": replica.port,
                "device_identifier": replica.device_identifier,
                "ready": replica.ready,
                "in_flight": replica.in_flight,
                "request_count": replica.request_count,
                "process_alive": process_alive,
                "pid": replica.process.pid if replica.process is not None else None,
                "exit_code": exit_code,
                "last_error": replica.last_error,
            }
        )

    backend_pid = details[0]["pid"] if details else None
    backend_exit_code = next((detail["exit_code"] for detail in details if detail["exit_code"] is not None), None)
    backend_process_alive = any(detail["process_alive"] for detail in details)
    backend_ready_count = sum(1 for detail in details if detail["ready"])
    return backend_pid, backend_exit_code, backend_process_alive, backend_ready_count, details


def get_health_payload() -> dict:
    if should_manage_backend_process():
        healthy, backend_health, message = _probe_backend_health()
        backend_pid: Optional[int] = None
        backend_exit_code: Optional[int] = None
        backend_process_alive = False
        backend_ready_count = 1 if healthy else 0
        backend_replica_details: list[dict[str, Any]] = []
        with _backend_lock:
            (
                backend_pid,
                backend_exit_code,
                backend_process_alive,
                backend_ready_count,
                backend_replica_details,
            ) = _managed_backend_runtime_locked()

        backend_state = _backend_state(healthy, backend_process_alive, backend_exit_code)
        backend_last_error = _backend_message(healthy, backend_process_alive, backend_exit_code, message)
        return {
            "status": "ok" if healthy else "degraded",
            "backend": "qwen-asr-workers",
            "backend_ready": healthy,
            "backend_ready_count": backend_ready_count,
            "backend_replica_count": len(backend_replica_details),
            "backend_state": backend_state,
            "backend_process_alive": backend_process_alive,
            "backend_urls": [detail["base_url"] for detail in backend_replica_details],
            "backend_pid": backend_pid,
            "backend_exit_code": backend_exit_code,
            "backend_last_error": backend_last_error,
            "backend_health": backend_health,
            "backend_replicas": backend_replica_details,
            "model_loaded": healthy,
            "model_id": _current_model_id,
            "revision": _current_model_revision,
            "device_map": "replicated",
            "dtype": DTYPE,
            "max_concurrent_transcribe": MAX_CONCURRENT_TRANSCRIBE,
            "max_backend_in_flight_per_replica": MAX_BACKEND_IN_FLIGHT_PER_REPLICA,
            "chunk_seconds": CHUNK_SECONDS,
            "chunk_overlap_seconds": CHUNK_OVERLAP_SECONDS,
            "context_tail_chars": CONTEXT_TAIL_CHARS,
            "normalize_zh_numbers": NORMALIZE_ZH_NUMBERS,
            "mcp_max_input_bytes": MCP_MAX_INPUT_BYTES,
            "backend_host": BACKEND_HOST,
            "backend_port": BACKEND_PORT,
            "auto_backend_replicas": _env_flag(AUTO_BACKEND_REPLICAS_ENV, "1"),
            "manage_backend_process": True,
            "started_at": _backend_started_at,
            "server_time": datetime.now().astimezone().isoformat(),
        }

    return {
        "status": "ok",
        "backend": "qwen-asr-local",
        "backend_ready": _model is not None,
        "backend_replica_count": 1,
        "model_loaded": _model is not None,
        "model_id": _current_model_id,
        "revision": _current_model_revision,
        "device_map": DEVICE_MAP,
        "dtype": DTYPE,
        "max_concurrent_transcribe": MAX_CONCURRENT_TRANSCRIBE,
        "max_backend_in_flight_per_replica": MAX_BACKEND_IN_FLIGHT_PER_REPLICA,
        "chunk_seconds": CHUNK_SECONDS,
        "chunk_overlap_seconds": CHUNK_OVERLAP_SECONDS,
        "context_tail_chars": CONTEXT_TAIL_CHARS,
        "normalize_zh_numbers": NORMALIZE_ZH_NUMBERS,
        "mcp_max_input_bytes": MCP_MAX_INPUT_BYTES,
    }
