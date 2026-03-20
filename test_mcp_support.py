import json
import unittest
from base64 import b64encode
from unittest.mock import AsyncMock, patch

from transcription_service import (
    InputTooLargeError,
    InputValidationError,
    decode_audio_base64,
    get_health_payload,
    guess_audio_suffix,
)


class TestTranscriptionHelpers(unittest.TestCase):
    def test_decode_audio_base64_success(self):
        payload = b64encode(b"demo-bytes").decode("ascii")
        self.assertEqual(decode_audio_base64(payload), b"demo-bytes")

    def test_decode_audio_base64_accepts_data_url(self):
        payload = "data:audio/wav;base64," + b64encode(b"wav").decode("ascii")
        self.assertEqual(decode_audio_base64(payload), b"wav")

    def test_decode_audio_base64_invalid(self):
        with self.assertRaises(InputValidationError):
            decode_audio_base64("%%%not-base64%%%")

    def test_decode_audio_base64_too_large(self):
        payload = b64encode(b"12345").decode("ascii")
        with self.assertRaises(InputTooLargeError):
            decode_audio_base64(payload, max_bytes=4)

    def test_guess_audio_suffix_prefers_filename_then_mime(self):
        self.assertEqual(guess_audio_suffix(filename="meeting.mp3"), ".mp3")
        self.assertEqual(guess_audio_suffix(mime_type="audio/wav"), ".wav")
        self.assertEqual(guess_audio_suffix(mime_type="video/mp4"), ".mp4")
        self.assertEqual(guess_audio_suffix(filename="bad.EXT!"), ".bin")


class TestMcpModule(unittest.IsolatedAsyncioTestCase):
    async def test_transcribe_audio_impl_success(self):
        from mcp_server import transcribe_audio_impl

        payload = b64encode(b"fake-audio").decode("ascii")
        with patch("mcp_server.transcribe_input_bytes", new=AsyncMock(return_value={"text": "ok", "language": "Chinese"})) as mocked:
            result = await transcribe_audio_impl(
                audio_base64=payload,
                filename="clip.mp3",
                language="zh",
                prompt="术语表",
            )

        self.assertEqual(result["text"], "ok")
        self.assertEqual(result["language"], "Chinese")
        mocked.assert_awaited_once()
        self.assertEqual(mocked.await_args.kwargs["suffix"], ".mp3")

    async def test_transcribe_audio_impl_backend_error(self):
        from mcp_server import transcribe_audio_impl
        from transcription_service import BackendTranscriptionError

        payload = b64encode(b"fake-audio").decode("ascii")
        with patch(
            "mcp_server.transcribe_input_bytes",
            new=AsyncMock(side_effect=BackendTranscriptionError("boom")),
        ):
            with self.assertRaisesRegex(RuntimeError, "transcribe_failed: boom"):
                await transcribe_audio_impl(audio_base64=payload, mime_type="audio/wav")

    def test_health_resource_matches_service_health(self):
        from mcp_server import build_health_resource_content

        self.assertEqual(json.loads(build_health_resource_content()), get_health_payload())


class TestHttpApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app import app

        cls._client_cm = TestClient(app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def test_fastapi_routes_and_mcp_mount(self):
        root_resp = self.client.get("/")
        self.assertEqual(root_resp.status_code, 200)

        health_resp = self.client.get("/health")
        self.assertEqual(health_resp.status_code, 200)
        self.assertIn("model_loaded", health_resp.json())

        mcp_resp = self.client.get("/mcp", follow_redirects=False)
        self.assertNotEqual(mcp_resp.status_code, 404)

    def test_transcriptions_route_uses_shared_service(self):
        with patch(
            "app.transcribe_input_bytes",
            new=AsyncMock(return_value={"text": "hello", "language": "Chinese"}),
        ) as mocked:
            response = self.client.post(
                "/v1/audio/transcriptions",
                files={"file": ("meeting.wav", b"RIFF....", "audio/wav")},
                data={"language": "zh", "prompt": "季度复盘"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "hello", "language": "Chinese"})
        mocked.assert_awaited_once()
        self.assertEqual(mocked.await_args.kwargs["suffix"], ".wav")


if __name__ == "__main__":
    unittest.main()
