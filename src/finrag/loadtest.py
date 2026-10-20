"""Closed-loop load test against a running search service.

Each of C concurrent clients sends one request, waits for the answer, and sends the next,
for a fixed duration, cycling through the 150 FinanceBench questions. Reported per level:
throughput, latency percentiles of successful requests, error rate, and the server's
resident memory read from its own ``/metrics``.

Closed-loop clients can't overload the server (they slow down when it does), so the
interesting numbers are where p95 starts to climb as concurrency rises.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np


async def _client(client, url: str, questions: list[str], params: dict, start: int, stride: int,
                  deadline: float, out: list) -> None:
    i = start
    while time.perf_counter() < deadline:
        q = questions[i % len(questions)]
        i += stride
        t0 = time.perf_counter()
        status = 0
        try:
            r = await client.get(url, params={"q": q, **params})
            status = r.status_code
        except Exception:
            status = -1
        out.append((time.perf_counter() - t0, status))


async def _memory_mb(client, base: str) -> float | None:
    try:
        text = (await client.get(f"{base}/metrics")).text
    except Exception:
        return None
    for line in text.splitlines():
        if line.startswith("process_resident_memory_bytes"):
            return round(float(line.split()[-1]) / 2**20, 1)
    return None


async def run_level(base: str, questions: list[str], concurrency: int, seconds: float,
                    params: dict, warmup: int = 20) -> dict:
    import httpx

    url = f"{base}/search"
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=10.0, limits=limits) as client:
        for q in questions[:warmup]:
            await client.get(url, params={"q": q, **params})
        out: list[tuple[float, int]] = []
        t0 = time.perf_counter()
        deadline = t0 + seconds
        await asyncio.gather(*(
            _client(client, url, questions, params, c, concurrency, deadline, out)
            for c in range(concurrency)))
        wall = time.perf_counter() - t0
        mem = await _memory_mb(client, base)
    ok = np.array([lat for lat, st in out if st == 200]) * 1000
    errors = sum(1 for _, st in out if st != 200)
    return {
        "concurrency": concurrency, "seconds": round(wall, 2), "requests": len(out),
        "errors": errors, "error_rate": round(errors / max(len(out), 1), 4),
        "throughput_rps": round(len(ok) / wall, 1),
        "p50_ms": round(float(np.percentile(ok, 50)), 2) if len(ok) else None,
        "p95_ms": round(float(np.percentile(ok, 95)), 2) if len(ok) else None,
        "p99_ms": round(float(np.percentile(ok, 99)), 2) if len(ok) else None,
        "server_rss_mb": mem,
    }


def run(base: str, questions: list[str], method: str, levels=(1, 2, 4, 8, 16, 32),
        seconds: float = 30.0) -> list[dict]:
    rows = []
    for c in levels:
        row = asyncio.run(run_level(base, questions, c, seconds, {"method": method}))
        row["method"] = method
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def save(rows: list[dict], meta: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"meta": meta, "levels": rows}, indent=1) + "\n")
