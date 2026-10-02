"""Build a local index, search evidence, and optionally ask Ollama."""
import argparse
import json

from rag import DATA_DIR, answer, build_index, load_index, save_index, search, source_url
from sample import sample_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question',nargs='?')
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--sample',action='store_true',help='Build the bundled example instead of an imported repository')
    parser.add_argument('--lexical',action='store_true',help='Skip embeddings and search with BM25 keywords')
    parser.add_argument('--answer',action='store_true',help='Generate an answer with installed Ollama models')
    args = parser.parse_args()
    if args.sample:
        index = sample_index(use_embeddings=not args.lexical)
        DATA_DIR.mkdir(exist_ok=True)
        from sample import sample_chunks
        (DATA_DIR / 'chunks.json').write_text(json.dumps(sample_chunks()),encoding='utf-8')
        (DATA_DIR / 'repository.json').write_text(json.dumps(index['repository']),encoding='utf-8')
        save_index(index)
    elif args.build:
        chunks = json.loads((DATA_DIR / 'chunks.json').read_text(encoding='utf-8'))
        metadata_path = DATA_DIR / 'repository.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
        index = build_index(chunks,lambda done,total:print(f'Embedded {done}/{total}',flush=True),
                            use_embeddings=not args.lexical,repository=metadata)
        save_index(index)
    else:
        index = load_index()
    if args.question:
        sources = search(index,args.question,method='lexical' if args.lexical else 'hybrid')
        for number,source in enumerate(sources,1):
            print(f"[{number}] {source['file_path']}:{source['start_line']}-{source['end_line']}")
            link = source_url(source,index['repository'])
            if link:
                print(link)
        if args.answer:
            print('\n'+answer(args.question,sources))
    else:
        print(f"Ready: {len(index['chunks'])} searchable pieces.")


if __name__ == '__main__':
    main()
