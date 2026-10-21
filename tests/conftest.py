from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "tests" / "fixtures" / "mini"


@pytest.fixture(scope="session")
def mini_dir(tmp_path_factory):
    """A copy of the fixture corpus, so caches written during tests don't land in the repo."""
    import shutil

    dest = tmp_path_factory.mktemp("mini")
    shutil.copytree(MINI, dest, dirs_exist_ok=True)
    return dest
