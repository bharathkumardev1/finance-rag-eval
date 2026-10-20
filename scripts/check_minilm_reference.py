"""Compare finrag's ONNX MiniLM against sentence-transformers on the same texts.

CI runs this where Hugging Face is reachable. It fails if any embedding's cosine similarity
with the reference falls below 0.9999, which would mean the tokenizer, pooling, or
normalization in finrag.dense drifted from what sentence-transformers does.
"""

import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

from finrag.dense import MiniLM

model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "models/all-MiniLM-L6-v2")
golden = json.loads(Path("tests/fixtures/minilm_golden.json").read_text())
texts = golden["texts"] + [
    "Consolidated Statements of Cash Flows (In millions) 2022 2021 2020 Net income 1,320 3,162",
    "PayPal Holdings, Inc. reported positive working capital of $1.6 billion as of FY2022 end.",
    " ".join(["revenue"] * 400),  # longer than 256 tokens: both sides truncate the same way
]
ours = MiniLM(model_dir).encode(texts)
ref = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
theirs = ref.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
cos = (ours * theirs).sum(axis=1)
print(json.dumps({"min_cosine": float(cos.min()), "per_text": [round(float(c), 6) for c in cos]}))
sys.exit(0 if cos.min() > 0.9999 else 1)
