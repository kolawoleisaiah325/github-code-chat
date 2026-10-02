"""Measure retrieval on a small authored example, separately from generated answers."""
import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from rag import generate_answer, search
from sample import sample_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-models', action='store_true', help='Also measure vector and hybrid retrieval with installed Ollama models')
    parser.add_argument('--answers', action='store_true', help='Record local AI answers for human review; requires --with-models')
    args = parser.parse_args()
    if args.answers and not args.with_models:
        parser.error('--answers requires --with-models')
    index = sample_index(use_embeddings=args.with_models)
    directory = Path(__file__).parent / 'evaluation'
    cases = json.loads((directory / 'questions.json').read_text(encoding='utf-8'))
    methods = ['lexical', 'vector', 'hybrid'] if args.with_models else ['lexical']
    results = {}
    for method in methods:
        rows = []
        for case in cases:
            started = time.perf_counter()
            sources = search(index, case['question'], limit=3, method=method)
            latency = round((time.perf_counter()-started)*1000, 3)
            files = [source['file_path'] for source in sources]
            expected = case['file']
            rank = files.index(expected)+1 if expected in files else None
            row = {**case, 'retrieved_files': files, 'rank': rank, 'retrieval_ms': latency}
            if args.answers and method == 'hybrid':
                answer = generate_answer(case['question'], sources)
                row['answer'] = {key:value for key,value in answer.items() if key != 'sources'}
            rows.append(row)
        answerable = [row for row in rows if row['file']]
        unknown = [row for row in rows if not row['file']]
        results[method] = {
            'answerable_questions': len(answerable), 'out_of_scope_questions': len(unknown),
            'file_hit_at_1': round(sum(row['rank']==1 for row in answerable)/len(answerable),4),
            'file_hit_at_3': round(sum(row['rank'] is not None for row in answerable)/len(answerable),4),
            'mrr_at_3': round(sum(1/row['rank'] if row['rank'] else 0 for row in answerable)/len(answerable),4),
            'out_of_scope_no_result': sum(not row['retrieved_files'] for row in unknown),
            'median_retrieval_ms': round(statistics.median(row['retrieval_ms'] for row in rows),3),
            'cases': rows,
        }
    report = {'evaluated_at':datetime.now(timezone.utc).isoformat(), 'corpus':'Bundled fictional shop: five small Python files',
              'source_fingerprint':index['source_fingerprint'], 'embedding_model':index['embedding_model'],
              'limitations':'Twenty authored questions on five authored files. Relevance is evaluated at file level, not passage or answer correctness. This is a sanity check, not independent evidence of performance on real repositories. Latency depends on hardware and model warm-up. Dense search always returns nearest neighbors, including for out-of-scope questions.',
              'results':results}
    output = directory / ('model-results.json' if args.with_models else 'baseline-results.json')
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    for method,result in results.items():
        print(f"{method}: hit@1={result['file_hit_at_1']:.1%}, hit@3={result['file_hit_at_3']:.1%}, MRR@3={result['mrr_at_3']}, median={result['median_retrieval_ms']}ms; out-of-scope no-result={result['out_of_scope_no_result']}/{len(unknown)}")
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
