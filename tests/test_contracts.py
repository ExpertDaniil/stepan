from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import threading
import time
import unittest

from agent.config import ROOT
from agent.contracts import Schemas
from agent.errors import AgentError
from tests.helpers import Harness


class ContractTests(unittest.TestCase):
    def test_all_original_contract_examples(self):
        schemas = Schemas()
        for cid in schemas.operations:
            for filename, suffix in [("request.json", "Request"), ("response.json", "Response"), ("error.json", "Response")]:
                with self.subTest(contract=cid, file=filename):
                    value = json.loads((ROOT / "contracts/examples" / cid / filename).read_text(encoding="utf-8"))
                    schemas.validate(value, cid + "_" + suffix)
        variants = json.loads((ROOT / "contracts/examples/variants-index.json").read_text(encoding="utf-8"))
        for variant in variants:
            value = json.loads((ROOT / "contracts" / variant["file"]).read_text(encoding="utf-8"))
            # The design index paths were relative to its root, now rooted at contracts/.
            schemas.validate(value, variant["schema"])

    def test_nonfinite_and_extra_fields_rejected(self):
        schemas = Schemas()
        request = deepcopy(schemas.operations["C12"]["request_example"])
        request["payload"]["plan"]["model"]["hyperparameters"]["C"] = float("nan")
        with self.assertRaises(AgentError):
            schemas.validate(request, "C12_Request")
        request = deepcopy(schemas.operations["C12"]["request_example"])
        request["payload"]["unannounced_field"] = 1
        with self.assertRaises(AgentError):
            schemas.validate(request, "C12_Request")

    def test_concurrent_idempotent_call_executes_once(self):
        h = Harness()
        try:
            lock, calls = threading.Lock(), []
            data = h.schemas.operations["C00"]["response_example"]["data"]
            payload = h.schemas.operations["C00"]["request_example"]["payload"]
            def handler(_):
                with lock:
                    calls.append(1)
                time.sleep(0.05)
                return data
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: h.bus.call("C00", payload, handler, key="same-run"), range(2)))
            self.assertEqual(len(calls), 1)
            self.assertEqual(results[0]["data"], results[1]["data"])
            self.assertNotEqual(results[0]["request_id"], results[1]["request_id"])
        finally:
            h.close()

    def test_reused_key_with_different_payload_fails(self):
        h = Harness()
        try:
            payload = deepcopy(h.schemas.operations["C00"]["request_example"]["payload"])
            data = h.schemas.operations["C00"]["response_example"]["data"]
            h.bus.call("C00", payload, lambda _: data, key="run")
            payload["max_workers"] = 1
            with self.assertRaises(AgentError) as caught:
                h.bus.call("C00", payload, lambda _: data, key="run")
            self.assertEqual(caught.exception.code, "IDEMPOTENCY_CONFLICT")
        finally:
            h.close()
