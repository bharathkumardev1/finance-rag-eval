"""Quality gate: fail when a benchmark result drops below its recorded baseline.

The baseline file lists configurations and the minimum value each metric must keep, minus a
tolerance. CI runs the fixture benchmark and this gate, so a change that quietly hurts
retrieval fails the build instead of shipping.
"""

from __future__ import annotations

import json
from pathlib import Path


def check(results_path: str | Path, baseline_path: str | Path) -> list[str]:
    results = json.loads(Path(results_path).read_text(encoding="utf8"))
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf8"))
    tol = float(baseline.get("tolerance", 0.0))
    failures = []
    for rule in baseline["rules"]:
        match = rule["match"]
        rows = [r for r in results["rows"] if all(r.get(k) == v for k, v in match.items())]
        if not rows:
            failures.append(f"no result row matches {match}")
            continue
        for metric, minimum in rule["min"].items():
            value = rows[0][metric]
            if value < minimum - tol:
                failures.append(
                    f"{match}: {metric}={value:.3f} is below baseline {minimum:.3f} "
                    f"(tolerance {tol:.3f})"
                )
    return failures
