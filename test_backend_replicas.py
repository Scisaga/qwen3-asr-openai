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


if __name__ == "__main__":
    unittest.main()
