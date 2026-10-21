import os
import zlib
from pathlib import Path

import numpy as np
import pytest

MODEL_DIR = Path(os.environ.get("FINRAG_MODEL_DIR", "models/all-MiniLM-L6-v2"))
needs_model = pytest.mark.skipif(not (MODEL_DIR / "onnx" / "model.onnx").exists(),
                                 reason="MiniLM not downloaded (finrag model)")


class HashEncoder:
    """Deterministic stand-in for MiniLM: hashed bag of words, L2-normalized."""

    name = "hash"
    dim = 64
    sha256 = "test"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for w in text.lower().split():
            v[zlib.crc32(w.encode()) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def encode_pages(self, pages):
        vecs, owner = [], []
        for i, p in enumerate(pages):
            words = p.split()
            for start in range(0, max(len(words), 1), 30):  # several windows per page
                vecs.append(self._vec(" ".join(words[start:start + 30])))
                owner.append(i)
        return np.stack(vecs), np.array(owner, dtype=np.int32)
