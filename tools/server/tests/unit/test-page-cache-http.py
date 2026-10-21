#!/usr/bin/env python3
"""Exercise CPU HTTP cache reuse, tmpfs restart, and SSD-only recovery.

Use the tiny fixture emitted by test-fork-llama-archs -a glm5next -o DIR.
All cache directories are temporary and isolated from deployment state.
"""
import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(
    description="CPU-only HTTP checkpoint restart regression using a synthetic GLM fixture"
)
parser.add_argument("--server", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(
    prefix="llama-public-ram-", dir="/dev/shm"
) as ram, tempfile.TemporaryDirectory(prefix="llama-public-ssd-") as disk:
    env = dict(
        os.environ,
        LLAMA_SERVER_PAGE_CACHE_RAM_DIR=ram,
        LLAMA_SERVER_PAGE_CACHE_SSD_DIR=disk,
        LLAMA_SERVER_PAGE_CACHE_NAMESPACE=hashlib.sha256(
            b"isolated-synthetic-public-port-test-v1"
        ).hexdigest(),
        LLAMA_SERVER_PAGE_CACHE_SSD_BYTES=str(64 << 20),
        LLAMA_SERVER_PAGE_CACHE_SSD_RESERVE_BYTES=str(1 << 20),
        LLAMA_SERVER_PAGE_CACHE_SSD_QUEUE_BYTES=str(8 << 20),
    )
    evidence = []
    for phase in range(3):
        if phase == 2:
            for entry in Path(ram).iterdir():
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        log = (args.output / f"server-{phase}.log").open("w")
        cmd = [
            str(args.server.resolve()),
            "--model",
            str(args.model.resolve()),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--n-gpu-layers",
            "0",
            "--ctx-size",
            "512",
            "--parallel",
            "1",
            "--threads",
            "4",
            "--threads-batch",
            "4",
            "--batch-size",
            "128",
            "--ubatch-size",
            "128",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
            "--flash-attn",
            "on",
            "--cache-ram",
            "0",
            "--cache-page-ram",
            "32",
            "--cache-page-tokens",
            "32",
            "--no-warmup",
            "--fit",
            "off",
            "--chat-template",
            "chatml",
        ]
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)

        def api(path, body=None):
            data = None if body is None else json.dumps(body).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}" + path,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=90) as f:
                return json.load(f)

        try:
            for _ in range(100):
                if p.poll() is not None:
                    raise RuntimeError(f"server exited {p.returncode}; phase {phase}")
                try:
                    if api("/health").get("status") == "ok":
                        break
                except (OSError, urllib.error.HTTPError):
                    pass
                time.sleep(0.2)
            else:
                raise RuntimeError("health timeout")
            prompt = list(range(10, 106))
            for attempt in range(2):
                response = api(
                    "/completion",
                    {
                        "prompt": prompt,
                        "n_predict": 1,
                        "temperature": 0,
                        "seed": 42,
                        "cache_prompt": True,
                        "cache_control": {
                            "priority": "interactive",
                            "lifecycle": "retained",
                            "branch": "synthetic-public-test",
                        },
                    },
                )
                probe = api("/v1/internal/cache-probe", {"prompt": prompt})
                evidence.append(
                    {
                        "phase": phase,
                        "attempt": attempt,
                        "content": response.get("content"),
                        "timings": response.get("timings"),
                        "probe": probe,
                    }
                )
            for _ in range(100):
                tiers = api("/v1/internal/cache-probe", {"prompt": prompt})[
                    "page_cache"
                ]["persistence"]["tiers"]
                if tiers["pending_bytes"] == 0 and (
                    tiers["used_bytes"] > 0 or tiers["written_bytes"] > 0
                ):
                    break
                time.sleep(0.05)
            else:
                raise AssertionError(("SSD backup did not complete", tiers))
        finally:
            p.terminate()
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            log.close()
    (args.output / "evidence.json").write_text(json.dumps(evidence, indent=2))
    assert evidence[0]["timings"]["cache_n"] == 0
    assert len({x["content"] for x in evidence}) == 1
    for x in evidence[1:]:
        assert x["timings"]["cache_n"] == 95, x
        assert x["probe"]["page_cache"]["failures"] == 0, x
    for x in (evidence[2], evidence[4]):
        page = x["probe"]["page_cache"]
        assert page["hits"] > 0 and page["persistence"]["recovered_pages"] > 0, x
    assert (
        evidence[4]["probe"]["page_cache"]["persistence"]["tiers"]["ssd_restores"] > 0
    )
    print(
        json.dumps(
            [
                {
                    "phase": x["phase"],
                    "attempt": x["attempt"],
                    "timings": x["timings"],
                    "cache_keys": list(x["probe"]),
                }
                for x in evidence
            ]
        )
    )
