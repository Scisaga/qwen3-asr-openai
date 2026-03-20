import json
from typing import Optional

from mcp.server.fastmcp import FastMCP

from transcription_service import (
    BackendTranscriptionError,
    CudaOOMTranscriptionError,
    MCP_MAX_INPUT_BYTES,
    decode_audio_base64,
    get_health_payload,
    guess_audio_suffix,
    transcribe_input_bytes,
)

mcp = FastMCP("Qwen3-ASR", stateless_http=True, json_response=True)
mcp.settings.streamable_http_path = "/"


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


def build_health_resource_content() -> str:
    return json.dumps(get_health_payload(), ensure_ascii=False, indent=2)


def build_usage_resource_content() -> str:
    return "\n".join(
        [
            "# Qwen3-ASR MCP Usage",
            "",
            "Use `transcribe_audio` when the client can only talk to the remote MCP server.",
            "",
            "Arguments:",
            "- `audio_base64` (required): base64 string or data URL containing the audio/video bytes",
            "- `filename` (optional): original filename, used to infer the temporary suffix",
            "- `mime_type` (optional): MIME type such as `audio/mpeg`, `audio/wav`, `video/mp4`",
            "- `language` (optional): same semantics as the HTTP API, e.g. `zh` or `en`",
            "- `prompt` (optional): domain context or proper nouns to bias transcription",
            "",
            f"Limits: decoded payload must be <= {_format_bytes(MCP_MAX_INPUT_BYTES)}.",
            "For large files, use `POST /v1/audio/transcriptions` instead of MCP base64 input.",
            "",
            'Return shape: `{"text": "...", "language": "Chinese"}`',
        ]
    )


async def transcribe_audio_impl(
    audio_base64: str,
    filename: Optional[str] = None,
    mime_type: Optional[str] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> dict:
    audio_bytes = decode_audio_base64(audio_base64)
    suffix = guess_audio_suffix(filename=filename, mime_type=mime_type)
    try:
        return await transcribe_input_bytes(
            audio_bytes,
            suffix=suffix,
            language=language,
            prompt=prompt,
        )
    except CudaOOMTranscriptionError as exc:
        raise RuntimeError(json.dumps(exc.detail, ensure_ascii=False)) from exc
    except BackendTranscriptionError as exc:
        raise RuntimeError(f"transcribe_failed: {exc}") from exc


@mcp.tool()
async def transcribe_audio(
    audio_base64: str,
    filename: Optional[str] = None,
    mime_type: Optional[str] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> dict:
    """Transcribe a base64-encoded audio or video payload."""
    return await transcribe_audio_impl(
        audio_base64=audio_base64,
        filename=filename,
        mime_type=mime_type,
        language=language,
        prompt=prompt,
    )


@mcp.resource("qwen3asr://health")
def qwen3asr_health() -> str:
    """Expose read-only runtime status for MCP clients."""
    return build_health_resource_content()


@mcp.resource("qwen3asr://usage")
def qwen3asr_usage() -> str:
    """Describe how to call the Qwen3-ASR MCP tool safely."""
    return build_usage_resource_content()


@mcp.prompt()
def transcribe_audio_workflow() -> str:
    """Guide the client to invoke the transcription tool correctly."""
    return (
        "When a user asks to transcribe audio or video and the file is available inline, "
        "call `transcribe_audio` with `audio_base64`. "
        "Pass `language` only if the user specifies it or it is strongly implied. "
        "Use `prompt` for names, jargon, or meeting context. "
        f"If the decoded payload would exceed {_format_bytes(MCP_MAX_INPUT_BYTES)}, "
        "tell the user to switch to the HTTP upload endpoint `POST /v1/audio/transcriptions`."
    )


@mcp.prompt()
def transcript_cleanup_workflow() -> str:
    """Guide the client to post-process transcripts without changing facts."""
    return (
        "After receiving a transcript, preserve factual content and speaker meaning. "
        "You may fix punctuation, split paragraphs, add headings, or convert it into notes. "
        "Do not fabricate missing words. "
        "If the transcript seems uncertain, keep the original wording visible and note ambiguity."
    )
