"""Lesson 2: split extracted files into overlapping pieces with source references."""

import argparse
import json
import re
from pathlib import Path


# GitIngest separates files with this header in its text output.
FILE_HEADER = re.compile(r"(?m)^={48}\nFILE: ([^\n]+)\n={48}\n")
DATA_DIR = Path(__file__).parent / "data"


def chunk_repository(content, max_lines=40, overlap=8):
    if not 0 <= overlap < max_lines:
        raise ValueError("Choose max_lines > 0 and 0 <= overlap < max_lines.")

    content = content.replace("\r\n", "\n")
    headers = list(FILE_HEADER.finditer(content))
    if not headers:
        raise ValueError("No GitIngest file headers found. Run read_repo.py first.")

    chunks = []
    for number, header in enumerate(headers):
        file_path = header.group(1)
        end = headers[number + 1].start() if number + 1 < len(headers) else len(content)
        # Remove the blank separators between files, while keeping leading lines.
        file_text = content[header.end():end].rstrip("\n")
        lines = file_text.splitlines(keepends=True)

        for start in range(0, len(lines), max_lines - overlap):
            stop = min(start + max_lines, len(lines))
            chunks.append({
                "id": f"{file_path}:{start + 1}-{stop}",
                "file_path": file_path,
                "start_line": start + 1,
                "end_line": stop,
                "text": "".join(lines[start:stop]),
            })
            if stop == len(lines):
                break

    return chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines", type=int, default=40, help="Maximum lines per chunk")
    parser.add_argument("--overlap", type=int, default=8, help="Lines shared by adjacent chunks")
    args = parser.parse_args()

    content = (DATA_DIR / "content.txt").read_text(encoding="utf-8")
    chunks = chunk_repository(content, args.lines, args.overlap)
    output = DATA_DIR / "chunks.json"
    output.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    file_count = len({chunk["file_path"] for chunk in chunks})
    print(f"Created {len(chunks)} chunks from {file_count} files.")
    print(f"Saved to: {output}")
    if chunks:
        print(f"First chunk: {chunks[0]['id']}")


if __name__ == "__main__":
    main()
