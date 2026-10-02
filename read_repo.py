"""Extract a public GitHub snapshot, or a user-selected local folder, as text."""
import argparse
import json
from pathlib import Path

from repository import ingest_repository


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Public GitHub repository root URL or local folder')
    args = parser.parse_args()
    if args.source.startswith('https://'):
        summary, tree, content, chunks, metadata = ingest_repository(args.source)
    else:
        from gitingest import ingest
        from chunk_repo import chunk_repository
        folder = Path(args.source).expanduser().resolve(strict=True)
        if not folder.is_dir():
            parser.error('Choose a local folder.')
        summary, tree, content = ingest(str(folder), max_file_size=100_000)
        chunks = chunk_repository(content)
        metadata = {'name': folder.name, 'kind': 'local folder'}
    output = Path(__file__).parent / 'data'
    output.mkdir(exist_ok=True)
    for name, text in [('summary.txt',summary), ('tree.txt',tree), ('content.txt',content)]:
        (output / name).write_text(text,encoding='utf-8')
    (output / 'chunks.json').write_text(json.dumps(chunks),encoding='utf-8')
    (output / 'repository.json').write_text(json.dumps(metadata),encoding='utf-8')
    print(summary)
    print(f'Extracted {len(chunks)} chunks into {output}. Build the index with search_repo.py --build.')


if __name__ == '__main__':
    main()
