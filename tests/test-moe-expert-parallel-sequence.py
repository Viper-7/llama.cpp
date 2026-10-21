#!/usr/bin/env python3
"""Fresh-process CPU coverage for the opt-in chronological fixture."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import shutil
import tempfile
import unittest


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def capture_text():
    events = [
        dict(kind="contract", version=1, mode="routes", warmup_excluded=True,
             single_slot=True, max_route_q=6),
        dict(kind="metadata", architecture="glm5next", experts=16, top_k=8,
             revision="chronological-cpu-fixture", model_file="synthetic"),
    ]
    # Same keys with changed IDs/coefficients/activations, then empty owners,
    # disappearing buckets, ragged cardinalities, old-key reuse, q3/q1 fallback.
    splits = [[4] * 6, [4] * 6, [8] * 6, [0] * 6,
              [4, 3, 2, 1, 0, 8], [4] * 6, [4] * 3, [4]]
    for index, counts in enumerate(splits):
        call, q = index + 1, len(counts)
        events.append(dict(kind="phase_begin", call=call, role="target", phase="target",
                           batch_q=q, prompt=False, replay=index == 5))
        ids, coeff = [], []
        for token, n in enumerate(counts):
            local = [(slot + index + token) % 8 for slot in range(n)]
            remote = [8 + (slot + index + token) % 8 for slot in range(8 - n)]
            selected = local + remote
            shift = (index + token) % 8
            ids.extend(selected[shift:] + selected[:shift])
            coeff.extend(f32((slot + 1 + index) / 64) for slot in range(8))
        for stage, dtype, values in (("ffn_moe_topk", "i32", ids),
                                     ("ffn_moe_weights", "f32", coeff)):
            events.append(dict(kind="route", call=call, role="target", phase="target",
                               layer=3, layer_call=1, q=q, top_k=8, stage=stage,
                               dtype=dtype, order="token_then_original_slot", values=values))
        events.append(dict(kind="phase_end", call=call, role="target", phase="target", status=0))
    text = ""
    for ordinal, event in enumerate(events, 1):
        text += json.dumps(dict(event, record=ordinal), separators=(",", ":")) + "\n"
    return text + json.dumps(dict(kind="final", status="request_complete", open_phase=False,
                                 records=len(events), bytes_before_final=len(text.encode()))) + "\n"


class SequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="moe-sequence-cpu-")
        cls.root = Path(cls.temp.name)
        cls.capture = cls.root / "routes.jsonl"
        cls.capture.write_text(capture_text())
        cls.reference = cls.root / "native.json"
        cls.command_index = 0
        cls.native = cls.invoke("native", "--reference-out", str(cls.reference))
        if cls.native.returncode:
            raise RuntimeError(cls.native.stderr)

    @classmethod
    def tearDownClass(cls):
        if OPTIONS.artifact_dir:
            root = Path(OPTIONS.artifact_dir) / "fixtures"
            root.mkdir(parents=True, exist_ok=True)
            for name in ("routes.jsonl", "native.json", "quant-native.json"):
                source = cls.root / name
                if source.is_file():
                    shutil.copy2(source, root / name)
        cls.temp.cleanup()

    @classmethod
    def invoke(cls, mode, *args, capture=None):
        command = [OPTIONS.binary, "--sequence", mode, "--route-capture", str(capture or cls.capture), *args]
        result = subprocess.run(command, text=True, capture_output=True,
                                env=dict(os.environ, HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES=""), timeout=90)
        if OPTIONS.artifact_dir:
            root = Path(OPTIONS.artifact_dir)
            root.mkdir(parents=True, exist_ok=True)
            prefix = root / f"command-{cls.command_index:03d}"
            prefix.with_suffix(".json").write_text(json.dumps(dict(command=command, exit=result.returncode), indent=2) + "\n")
            prefix.with_suffix(".stdout.jsonl").write_text(result.stdout)
            prefix.with_suffix(".stderr.log").write_text(result.stderr)
        cls.command_index += 1
        return result

    def records(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return [json.loads(line) for line in result.stdout.splitlines()]

    def test_chronology_and_stale_data(self):
        for mode in ("native", "owner", "bucket"):
            with self.subTest(mode=mode):
                result = self.native if mode == "native" else self.invoke(mode, "--reference-in", str(self.reference))
                records = self.records(result)
                self.assertFalse(any(row["kind"] == "sample_order_contract" for row in records))
                contract = next(row for row in records if row["kind"] == "sequence_contract")
                dimension = contract["reference_identity"]["D"]
                samples = [row for row in records if row["kind"] == "sequence_sample"]
                self.assertEqual(len(samples), 24)
                self.assertEqual([(row["pass"], row["capture_index"]) for row in samples],
                                 [(p, i) for p in range(3) for i in range(8)])
                for row in samples:
                    self.assertLessEqual(row["nmse"], 1e-10)
                    self.assertGreaterEqual(row["routed_region_plus_all_preparation_ms"], 0)
                    if row["q"] < 6 or mode == "native":
                        self.assertEqual(row["route_D2H_bytes"], 0)
                        self.assertEqual(row["keys"], [])
                        self.assertEqual(row["initial_zero_tensor_upload_bytes"], 0)
                        self.assertEqual(row["inactive_clear_copies"], 0)
                        self.assertEqual(row["inactive_clear_same_backend_bytes"], 0)
                    else:
                        slot_count = 16 if mode == "bucket" else 2
                        cleared = slot_count - len(row["keys"])
                        self.assertEqual(row["inactive_clear_copies"], cleared)
                        self.assertEqual(row["inactive_clear_same_backend_bytes"], cleared * dimension * row["q"] * 4)
                        self.assertEqual(row["initial_zero_tensor_upload_bytes"],
                                         2 * dimension * row["q"] * 4 if row["q_shape_miss"] else 0)
                if mode != "native":
                    self.assertTrue(all(key["host_hit"] for key in samples[1]["keys"]))
                    self.assertTrue(all(key["host_hit"] for row in samples[8:] for key in row["keys"]))
                    self.assertEqual({key["owner"] for key in samples[2]["keys"]}, {0})
                    self.assertEqual({key["owner"] for key in samples[3]["keys"]}, {1})
                self.assertEqual(sum(row["q_shape_miss"] for row in samples), 3)

    def test_caps_fail_before_allocation(self):
        for option in ("--cache-bytes", "--cache-entries"):
            result = self.invoke("bucket", "--reference-in", str(self.reference), option, "1")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("cap exceeded before allocation", result.stderr)

    def test_quantized_modes(self):
        reference = self.root / "quant-native.json"
        result = self.invoke("native", "--quant", "--reference-out", str(reference))
        self.records(result)
        for mode in ("owner", "bucket"):
            records = self.records(self.invoke(mode, "--quant", "--reference-in", str(reference)))
            samples = [row for row in records if row["kind"] == "sequence_sample"]
            self.assertEqual(len(samples), 24)
            self.assertTrue(all(row["nmse"] <= 5e-4 for row in samples))

    def test_reference_exclusive_and_invalid_contracts(self):
        before = self.reference.read_bytes()
        result = self.invoke("native", "--reference-out", str(self.reference))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.reference.read_bytes(), before)
        for args in (("--warmups", "0"), ("--repetitions", "3"), ("--route-case", "0")):
            result = self.invoke("bucket", "--reference-in", str(self.reference), *args)
            self.assertNotEqual(result.returncode, 0)

    def test_reference_provenance_and_payload(self):
        original = json.loads(self.reference.read_text())
        variants = []
        for field in ("capture_sha256", "bank_bytes_sha256_gate_up_down", "binary"):
            doc = json.loads(json.dumps(original))
            doc["identity"][field] = "wrong"
            variants.append(json.dumps(doc))
        doc = json.loads(json.dumps(original)); doc["outputs"][0][0] = 0x7FC00000
        variants.append(json.dumps(doc))
        doc = json.loads(json.dumps(original)); doc["outputs"][0].pop()
        variants.append(json.dumps(doc))
        doc = json.loads(json.dumps(original)); doc["complete"] = False
        variants.append(json.dumps(doc))
        variants.extend([json.dumps(original) + "trailing", "{", "[]"])
        variants.append("[" * 100 + "0" + "]" * 100)
        for i, data in enumerate(variants):
            with self.subTest(i=i):
                path = self.root / f"bad-{i}.json"; path.write_text(data)
                result = self.invoke("bucket", "--reference-in", str(path))
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('"kind":"sequence_sample"', result.stdout)

    def test_reference_oversize(self):
        path = self.root / "oversized.json"
        with path.open("wb") as output:
            output.seek(128 * 1024 * 1024)
            output.write(b"\n")
        result = self.invoke("bucket", "--reference-in", str(path))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("reference input unreadable/oversized", result.stderr)

    def test_incomplete_capture(self):
        path = self.root / "incomplete.jsonl"
        path.write_text("\n".join(capture_text().splitlines()[:-1]) + "\n")
        result = self.invoke("bucket", "--reference-in", str(self.reference), capture=path)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('"kind":"sequence_sample"', result.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("binary")
    parser.add_argument("--artifact-dir")
    OPTIONS = parser.parse_args()
    unittest.main(argv=[__file__], verbosity=2)
