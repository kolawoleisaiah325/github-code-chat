"""Lesson 3: build an embedding index, search it, and optionally ask Llama."""

import argparse
import json

from rag import DATA_DIR, answer, build_index, load_index, save_index, search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--answer", action="store_true")
    args = parser.parse_args()

    if args.build:
        chunks = json.loads((DATA_DIR / "chunks.json").read_text(encoding="utf-8"))
        index = build_index(chunks, lambda done, total: print(f"Embedded {done}/{total}", flush=True))
        save_index(index)
        print("Saved the searchable index.")
    else:
        index = load_index()

    if args.question:
        sources = search(index, args.question)
        for number, source in enumerate(sources, 1):
            print(f"[{number}] {source['file_path']}:{source['start_line']}-{source['end_line']} "
                  f"(similarity {source['score']:.3f})")
        if args.answer:
            print("\n" + answer(args.question, sources))


if __name__ == "__main__":
    main()
