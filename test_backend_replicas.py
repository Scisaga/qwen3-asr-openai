import asyncio
import os
import unittest
from unittest.mock import patch

import transcription_service as svc


class TestBackendReplicaConfig(unittest.TestCase):
    def test_auto_uses_one_replica_per_visible_gpu(self):
        with patch.dict(os.environ, {}, clear=False):
            for key in (
                svc.ASR_BACKEND_WORKER_ENV,
                svc.MANAGE_BACKEND_PROCESS_ENV,
                svc.BACKEND_REPLICA_COUNT_ENV,
            ):
                os.environ.pop(key, None)
            os.environ[svc.AUTO_BACKEND_REPLICAS_ENV] = "1"

            with patch.object(svc, "_detect_visible_gpu_identifiers", return_value=["0", "1"]):
                self.assertTrue(svc.should_manage_backend_process())
                layout = svc._build_backend_replicas_layout()

        self.assertEqual([replica.device_identifier for replica in layout], ["0", "1"])
        self.assertEqual([replica.port for replica in layout], [svc.BACKEND_PORT, svc.BACKEND_PORT + 1])

    def test_worker_env_remaps_cuda_device_map_to_local_gpu(self):
        replica = svc.BackendReplica(
            replica_index=0,
            port=svc.BACKEND_PORT,
            base_url=f"http://{svc.BACKEND_HOST}:{svc.BACKEND_PORT}",
            device_identifier="1",
        )

        with patch.dict(os.environ, {"DEVICE_MAP": "cuda:1"}, clear=False):
            env = svc._build_backend_env(replica)

        self.assertEqual(env[svc.ASR_BACKEND_WORKER_ENV], "1")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["DEVICE_MAP"], "cuda:0")

    def test_router_prefers_least_busy_replica(self):
        replica0 = svc.BackendReplica(
            replica_index=0,
            port=svc.BACKEND_PORT,
            base_url=f"http://{svc.BACKEND_HOST}:{svc.BACKEND_PORT}",
            ready=True,
            in_flight=2,
        )
        replica1 = svc.BackendReplica(
            replica_index=1,
            port=svc.BACKEND_PORT + 1,
            base_url=f"http://{svc.BACKEND_HOST}:{svc.BACKEND_PORT + 1}",
            ready=True,
            in_flight=0,
        )
        original_replicas = svc._backend_replicas
        original_router_index = svc._backend_router_index
        try:
            svc._backend_replicas = [replica0, replica1]
            svc._backend_router_index = 0

            selected = svc._reserve_backend_replica_locked(set())

            self.assertIs(selected, replica1)
            self.assertEqual(replica1.in_flight, 1)
            self.assertEqual(replica1.request_count, 1)
        finally:
            svc._backend_replicas = original_replicas
            svc._backend_router_index = original_router_index

    def test_router_round_robins_when_load_is_equal(self):
        replica0 = svc.BackendReplica(
            replica_index=0,
            port=svc.BACKEND_PORT,
            base_url=f"http://{svc.BACKEND_HOST}:{svc.BACKEND_PORT}",
            ready=True,
        )
        replica1 = svc.BackendReplica(
            replica_index=1,
            port=svc.BACKEND_PORT + 1,
            base_url=f"http://{svc.BACKEND_HOST}:{svc.BACKEND_PORT + 1}",
            ready=True,
        )
        original_replicas = svc._backend_replicas
        original_router_index = svc._backend_router_index
        try:
            svc._backend_replicas = [replica0, replica1]
            svc._backend_router_index = 0

            selected0 = svc._reserve_backend_replica_locked(set())
            svc._release_backend_replica_locked(selected0, ready=True)
            selected1 = svc._reserve_backend_replica_locked(set())

            self.assertIs(selected0, replica0)
            self.assertIs(selected1, replica1)
        finally:
            svc._backend_replicas = original_replicas
            svc._backend_router_index = original_router_index

    def test_router_rejects_when_all_replicas_are_at_capacity(self):
        replica0 = svc.BackendReplica(
            replica_index=0,
            port=svc.BACKEND_PORT,
            base_url=f"http://{svc.BACKEND_HOST}:{svc.BACKEND_PORT}",
            ready=True,
            in_flight=svc.MAX_BACKEND_IN_FLIGHT_PER_REPLICA,
        )
        replica1 = svc.BackendReplica(
            replica_index=1,
            port=svc.BACKEND_PORT + 1,
            base_url=f"http://{svc.BACKEND_HOST}:{svc.BACKEND_PORT + 1}",
            ready=True,
            in_flight=svc.MAX_BACKEND_IN_FLIGHT_PER_REPLICA,
        )
        original_replicas = svc._backend_replicas
        original_router_index = svc._backend_router_index
        try:
            svc._backend_replicas = [replica0, replica1]
            svc._backend_router_index = 0

            selected = svc._reserve_backend_replica_locked(set())

            self.assertIsNone(selected)
            self.assertEqual(replica0.in_flight, svc.MAX_BACKEND_IN_FLIGHT_PER_REPLICA)
            self.assertEqual(replica1.in_flight, svc.MAX_BACKEND_IN_FLIGHT_PER_REPLICA)
        finally:
            svc._backend_replicas = original_replicas
            svc._backend_router_index = original_router_index


class TestBackendReplicaQueue(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.original_replicas = svc._backend_replicas
        self.original_router_index = svc._backend_router_index
        self.original_queue_waiters = svc._backend_queue_waiters
        self.original_queue_timeout = svc.BACKEND_QUEUE_TIMEOUT_SECONDS
        self.original_queue_poll_interval = svc.BACKEND_QUEUE_POLL_INTERVAL_SECONDS
        self.original_max_queue_requests = svc.MAX_BACKEND_QUEUE_REQUESTS

    async def asyncTearDown(self):
        svc._backend_replicas = self.original_replicas
        svc._backend_router_index = self.original_router_index
        svc._backend_queue_waiters = self.original_queue_waiters
        svc.BACKEND_QUEUE_TIMEOUT_SECONDS = self.original_queue_timeout
        svc.BACKEND_QUEUE_POLL_INTERVAL_SECONDS = self.original_queue_poll_interval
        svc.MAX_BACKEND_QUEUE_REQUESTS = self.original_max_queue_requests

    async def test_router_waits_for_backend_capacity(self):
        replica = svc.BackendReplica(
            replica_index=0,
            port=svc.BACKEND_PORT,
            base_url=f"http://{svc.BACKEND_HOST}:{svc.BACKEND_PORT}",
            ready=True,
            in_flight=svc.MAX_BACKEND_IN_FLIGHT_PER_REPLICA,
        )
        svc._backend_replicas = [replica]
        svc._backend_router_index = 0
        svc._backend_queue_waiters = 0
        svc.BACKEND_QUEUE_TIMEOUT_SECONDS = 0.2
        svc.BACKEND_QUEUE_POLL_INTERVAL_SECONDS = 0.01
        svc.MAX_BACKEND_QUEUE_REQUESTS = 1

        async def release_capacity():
            await asyncio.sleep(0.03)
            with svc._backend_lock:
                replica.in_flight = 0

        releaser = asyncio.create_task(release_capacity())
        selected = await svc._reserve_backend_replica_with_wait(set())
        await releaser

        self.assertIs(selected, replica)
        self.assertEqual(replica.in_flight, 1)
        self.assertEqual(svc._backend_queue_waiters, 0)

    async def test_router_rejects_when_wait_queue_is_full(self):
        replica = svc.BackendReplica(
            replica_index=0,
            port=svc.BACKEND_PORT,
            base_url=f"http://{svc.BACKEND_HOST}:{svc.BACKEND_PORT}",
            ready=True,
            in_flight=svc.MAX_BACKEND_IN_FLIGHT_PER_REPLICA,
        )
        svc._backend_replicas = [replica]
        svc._backend_router_index = 0
        svc._backend_queue_waiters = 1
        svc.BACKEND_QUEUE_TIMEOUT_SECONDS = 0.2
        svc.MAX_BACKEND_QUEUE_REQUESTS = 1

        selected = await svc._reserve_backend_replica_with_wait(set())

        self.assertIsNone(selected)
        self.assertEqual(svc._backend_queue_waiters, 1)

    async def test_cancelled_client_keeps_replica_reserved_until_backend_finishes(self):
        replica = svc.BackendReplica(
            replica_index=0,
            port=svc.BACKEND_PORT,
            base_url=f"http://{svc.BACKEND_HOST}:{svc.BACKEND_PORT}",
            ready=True,
        )
        svc._backend_replicas = [replica]
        svc._backend_router_index = 0
        started = asyncio.Event()
        finish = asyncio.Event()

        async def fake_post(*args, **kwargs):
            del args, kwargs
            started.set()
            await finish.wait()
            return {"text": "ok"}

        async def fake_ensure_backend_started(*args, **kwargs):
            del args, kwargs

        with patch.object(svc, "ensure_backend_started", fake_ensure_backend_started), patch.object(
            svc, "_post_transcription_to_backend", fake_post
        ):
            task = asyncio.create_task(svc._transcribe_input_bytes_via_backend(b"audio", suffix=".wav"))
            await started.wait()
            self.assertEqual(replica.in_flight, 1)

            task.cancel()
            await asyncio.sleep(0)
            self.assertEqual(replica.in_flight, 1)

            finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(replica.in_flight, 0)


if __name__ == "__main__":
    unittest.main()
