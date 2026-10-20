"""Quality gate: compare an eval.json against minimums in configs/gate.json.

CI runs the full pipeline on the synthetic fixture and fails the build if a tracked metric
falls more than ``tolerance`` below its minimum, so a change that breaks the filter, the
routing, or a retriever is caught before it is merged.
"""

from __future__ import annotations

import json
from pathlib import Path


def check(results_path: Path, baseline_path: Path) -> list[str]:
    results = json.loads(Path(results_path).read_text(encoding="utf8"))
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf8"))
    tol = float(baseline.get("tolerance", 0.0))
    rows = results.get("table", [])
    failures = []
    for rule in baseline["rules"]:
        match = rule["match"]
        found = [r for r in rows if all(r.get(k) == v for k, v in match.items())]
        if not found:
            failures.append(f"no results row matches {match}")
            continue
        for metric, minimum in rule["min"].items():
            got = found[0].get(metric)
            if got is None or got < minimum - tol:
                failures.append(f"{match}: {metric}={got} below minimum {minimum} (tolerance {tol})")
    return failures
