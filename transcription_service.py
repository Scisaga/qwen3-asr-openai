import asyncio
import base64
import binascii
import math
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional

from text_normalize import normalize_zh_numbers

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-1.7B")
MODEL_REVISION = os.getenv("MODEL_REVISION")
DEVICE_MAP = os.getenv("DEVICE_MAP", "cuda:0")
DTYPE = os.getenv("DTYPE", "bfloat16")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "1"))
MAX_CONCURRENT_TRANSCRIBE = int(os.getenv("MAX_CONCURRENT_TRANSCRIBE", "1"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
CHUNK_SECONDS = float(os.getenv("CHUNK_SECONDS", "600"))
CHUNK_OVERLAP_SECONDS = float(os.getenv("CHUNK_OVERLAP_SECONDS", "1"))
CONTEXT_TAIL_CHARS = int(os.getenv("CONTEXT_TAIL_CHARS", "200"))
NORMALIZE_ZH_NUMBERS = (
    os.getenv("NORMALIZE_ZH_NUMBERS", "1").strip().lower()
    not in ("0", "false", "no", "off")
)
MCP_MAX_INPUT_BYTES = int(os.getenv("MCP_MAX_INPUT_BYTES", str(32 * 1024 * 1024)))

_model = None
_current_model_id = MODEL_ID
_current_model_revision = MODEL_REVISION
_transcribe_sem = asyncio.Semaphore(max(1, MAX_CONCURRENT_TRANSCRIBE))

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


class BackendTranscriptionError(RuntimeError):
    pass


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


async def transcribe_input_bytes(
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
                merged_text, merged_lang = await asyncio.to_thread(
                    _transcribe_path, input_path, mapped_language, context
                )
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


def get_health_payload() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_id": _current_model_id,
        "revision": _current_model_revision,
        "device_map": DEVICE_MAP,
        "dtype": DTYPE,
        "max_concurrent_transcribe": MAX_CONCURRENT_TRANSCRIBE,
        "chunk_seconds": CHUNK_SECONDS,
        "chunk_overlap_seconds": CHUNK_OVERLAP_SECONDS,
        "context_tail_chars": CONTEXT_TAIL_CHARS,
        "normalize_zh_numbers": NORMALIZE_ZH_NUMBERS,
        "mcp_max_input_bytes": MCP_MAX_INPUT_BYTES,
    }
