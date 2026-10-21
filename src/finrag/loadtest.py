"""Closed-loop load test against a running search service.

Each of C concurrent clients sends a request, waits for the answer and sends the next, for a
fixed time, cycling through the 150 FinanceBench questions. Per level it reports throughput,
latency percentiles of successful requests, the error rate and the server's resident memory
from its own ``/metrics``.

Closed-loop clients can't overload a server (they slow down when it does), so what matters
is where p95 starts to climb as concurrency rises, and the throughput there.

Cost per million queries prices one vCPU at half the on-demand rate of an AWS c7i.large
(2 vCPUs, $0.08925 an hour in us-east-1 on 2026-10-21, from instances.vantage.sh and
calculator.holori.com). It assumes the throughput measured here carries over to that vCPU,
which I haven't measured.
"""

from __future__ import annotations

import asyncio
import json
import time

import numpy as np

USD_PER_VCPU_HOUR = 0.08925 / 2


async def _client(client, url: str, questions: list[str], params: dict, start: int, stride: int,
                  deadline: float, out: list) -> None:
    i = start
    while time.perf_counter() < deadline:
        q = questions[i % len(questions)]
        i += stride
        t0 = time.perf_counter()
        try:
            status = (await client.get(url, params={"q": q, **params})).status_code
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


async def run_level(base: str, questions: list[str], concurrency: int, seconds: float, params: dict,
                    warmup: int = 20) -> dict:
    import httpx

    url = f"{base}/search"
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=10.0, limits=limits) as client:
        for q in questions[:warmup]:
            await client.get(url, params={"q": q, **params})
        out: list[tuple[float, int]] = []
        t0 = time.perf_counter()
        await asyncio.gather(*(_client(client, url, questions, params, c, concurrency, t0 + seconds, out)
                               for c in range(concurrency)))
        wall = time.perf_counter() - t0
        mem = await _memory_mb(client, base)
    ok = np.array([lat for lat, st in out if st == 200]) * 1000
    errors = sum(1 for _, st in out if st != 200)
    rps = len(ok) / wall
    return {
        "concurrency": concurrency, "seconds": round(wall, 2), "requests": len(out), "errors": errors,
        "error_rate": round(errors / max(len(out), 1), 4), "throughput_rps": round(rps, 1),
        "p50_ms": round(float(np.percentile(ok, 50)), 2) if len(ok) else None,
        "p95_ms": round(float(np.percentile(ok, 95)), 2) if len(ok) else None,
        "p99_ms": round(float(np.percentile(ok, 99)), 2) if len(ok) else None,
        "server_rss_mb": mem,
        "usd_per_million": round(USD_PER_VCPU_HOUR / (rps * 3600) * 1e6, 3) if rps else None,
    }


def run(base: str, questions: list[str], method: str, levels=(1, 2, 4, 8, 16), seconds: float = 30.0,
        params: dict | None = None) -> list[dict]:
    rows = []
    for c in levels:
        row = asyncio.run(run_level(base, questions, c, seconds, {"method": method, **(params or {})}))
        row["method"] = method
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows
