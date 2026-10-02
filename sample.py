"""Build a bundled demonstration index without networking or model downloads."""
from pathlib import Path

from rag import build_index

SAMPLE_DIR = Path(__file__).parent / "samples" / "shop"
SAMPLE_QUESTIONS = [
    "How does the WELCOME10 coupon change the order total?",
    "When is domestic shipping free?",
    "How does the shop reject a cart with insufficient stock?",
    "Can an order be cancelled after it has been marked paid?",
]


def sample_chunks():
    chunks = []
    for path in sorted(SAMPLE_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        lines = len(text.splitlines())
        chunks.append({"id": f"{path.name}:1-{lines}", "file_path": path.name,
                       "start_line": 1, "end_line": lines, "text": text})
    return chunks


def sample_index(use_embeddings=False):
    return build_index(sample_chunks(), use_embeddings=use_embeddings, repository={
        "name": "Demo Shop", "kind": "bundled example",
        "url": "https://github.com/kolawoleisaiah325/github-code-chat",
        "path_prefix": "samples/shop/", "ref": "main",
    })
