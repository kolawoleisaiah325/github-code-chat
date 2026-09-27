"""Lesson 1: extract a repository into text that we can later search."""

import argparse
from pathlib import Path

from gitingest import ingest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Public GitHub URL or local folder")
    args = parser.parse_args()

    print("Reading repository...", flush=True)
    summary, tree, content = ingest(args.source)

    output = Path(__file__).parent / "data"
    output.mkdir(exist_ok=True)
    for name, text in (("summary.txt", summary), ("tree.txt", tree), ("content.txt", content)):
        (output / name).write_text(text, encoding="utf-8")

    print(summary)
    print(f"\nExtracted text saved in: {output}")


if __name__ == "__main__":
    main()
