"""The scan exception stays as narrow as the comment in .gitleaks.toml says."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_exception_needs_rule_path_and_a_bare_hex_value():
    allow = tomllib.loads((ROOT / ".gitleaks.toml").read_text())["allowlists"][0]
    assert allow["condition"] == "AND" and allow["targetRules"] == ["generic-api-key"]
    path = re.compile(allow["paths"][0])
    assert path.search("results/generation/calls.jsonl") and path.search("results/rerank.json")
    assert path.search("results/rerank-windows3.json") and not path.search("results/rerank-notes.json")
    assert not path.search("src/finrag/cli.py") and not path.search("results/generation/labels.csv")
    value = re.compile(allow["regexes"][0])
    assert value.search("a" * 64) and not value.search("sk-" + "a" * 40)


def test_ci_scans_with_that_config():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "GITLEAKS_CONFIG: .gitleaks.toml" in ci


def test_ci_pins_a_gitleaks_that_understands_the_allowlist():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    pinned = re.search(r"GITLEAKS_VERSION: (\d+)\.(\d+)\.(\d+)", ci)
    assert pinned and tuple(map(int, pinned.groups())) >= (8, 25, 0)  # AND allowlists need 8.25
