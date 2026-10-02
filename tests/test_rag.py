import math
import unittest
from unittest.mock import patch

from chunk_repo import chunk_repository
from rag import (bounded_chunks, build_index, citation_check, context_sources,
                 cosine_similarity, generate_answer, search, source_url, validate_index)
from repository import validate_repository_url
from sample import sample_index


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = sample_index()

    def test_sample_works_without_ollama(self):
        with patch('rag.ollama_request', side_effect=AssertionError('Offline mode contacted Ollama')):
            matches = search(self.index, 'When is domestic shipping free?', method='lexical')
        self.assertEqual(matches[0]['file_path'], 'shipping.py')
        self.assertEqual(self.index['vectors'], [])

    def test_unknown_topic_has_no_keyword_evidence(self):
        self.assertEqual(search(self.index, 'Kubernetes replica autoscaling', method='lexical'), [])

    def test_hybrid_fuses_identifier_and_semantic_evidence(self):
        chunks = [{'id': str(i), 'file_path': f'{i}.py', 'start_line': 1, 'end_line': 1, 'text': text}
                  for i, text in enumerate(['handle_coupon discount', 'send_mail message', 'reserve stock'])]
        with patch('rag.embeddings', return_value=[[1, 0], [0, 1], [.5, .5]]):
            index = build_index(chunks)
        with patch('rag.embeddings', return_value=[[1, 0]]):
            matches = search(index, 'handle_coupon', method='hybrid')
        self.assertEqual(matches[0]['file_path'], '0.py')
        self.assertIsNotNone(matches[0]['vector_score'])

    def test_vector_mode_requires_embeddings(self):
        with self.assertRaises(ValueError):
            search(self.index, 'shipping', method='vector')

    def test_invalid_questions_rejected(self):
        for question in ['', 'x' * 2001]:
            with self.assertRaises(ValueError):
                search(self.index, question)

    def test_nonfinite_vectors_rejected(self):
        index = dict(self.index, embedding_model='nomic-embed-text', vectors=[[math.nan]] * len(self.index['chunks']))
        with self.assertRaises(ValueError):
            validate_index(index)

    def test_modified_source_invalidates_saved_index(self):
        self.index['chunks'][0]['text'] += '# changed'
        with self.assertRaises(ValueError):
            validate_index(self.index)

    def test_old_index_explains_rebuild(self):
        with self.assertRaisesRegex(ValueError, 'older format'):
            validate_index({'chunks': [], 'vectors': []})

    def test_source_url_uses_snapshot_commit_and_line_numbers(self):
        source = {'file_path': 'a folder/file.py', 'start_line': 3, 'end_line': 8}
        repo = {'url': 'https://github.com/owner/repo', 'commit': 'a' * 40}
        self.assertEqual(source_url(source, repo), f"https://github.com/owner/repo/blob/{'a'*40}/a%20folder/file.py#L3-L8")
        source['file_path'] = '../secret.py'
        self.assertIsNone(source_url(source, repo))

    def test_line_chunk_boundaries(self):
        lines = [f'line {i}\n' for i in range(1, 91)]
        digest = '=' * 48 + '\nFILE: example.py\n' + '=' * 48 + '\n' + ''.join(lines)
        chunks = chunk_repository(digest)
        self.assertEqual([(c['start_line'], c['end_line']) for c in chunks], [(1, 40), (33, 72), (65, 90)])
        for chunk in chunks:
            self.assertEqual(chunk['text'].rstrip('\n'), ''.join(lines[chunk['start_line']-1:chunk['end_line']]).rstrip('\n'))

    def test_long_line_and_lockfile_handling(self):
        chunk = {'id': 'x', 'file_path': 'x.py', 'start_line': 1, 'end_line': 1, 'text': 'x' * 4000}
        bounded = bounded_chunks([chunk, {**chunk, 'file_path': 'package-lock.json'}])
        self.assertEqual(len(bounded), 3)
        self.assertTrue(all(c['start_line'] == 1 and c['end_line'] == 1 for c in bounded))
        self.assertEqual(''.join(c['text'] for c in bounded), chunk['text'])

    def test_context_budget_preserves_whole_excerpts(self):
        chunks = [{**self.index['chunks'][0], 'text': 'x' * 1500} for _ in range(20)]
        chosen = context_sources(chunks)
        self.assertLess(len(chosen), len(chunks))
        self.assertTrue(all(c['text'] == 'x' * 1500 for c in chosen))

    def test_cosine_zero_vector(self):
        self.assertEqual(cosine_similarity([0, 0], [1, 0]), 0)
        with self.assertRaises(ValueError):
            cosine_similarity([1], [1, 2])


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.sources = search(sample_index(), 'WELCOME10 coupon', method='lexical')

    def test_no_evidence_does_not_call_model(self):
        with patch('rag.ollama_request', side_effect=AssertionError):
            self.assertEqual(generate_answer('unknown', [])['status'], 'no_evidence')

    def test_unknown_citation_rejects_generated_answer(self):
        with patch('rag.ollama_request', return_value={'message': {'content': 'It gives a discount [99].'}}):
            result = generate_answer('coupon', self.sources)
        self.assertEqual(result['status'], 'citation_failed')
        self.assertNotIn('It gives a discount', result['text'])
        self.assertTrue(result['sources'])

    def test_no_citation_rejects_generated_answer(self):
        self.assertFalse(citation_check('Unreferenced claim.', 2)['valid'])
        self.assertTrue(citation_check('Supported statement [1].', 2)['valid'])
        self.assertFalse(citation_check('Claim [0].', 2)['valid'])

    def test_ollama_failure_retains_sources(self):
        with patch('rag.ollama_request', side_effect=RuntimeError('Model unavailable')):
            result = generate_answer('coupon', self.sources)
        self.assertEqual(result['status'], 'unavailable')
        self.assertTrue(result['sources'])

    def test_valid_reference_is_format_checked_not_fact_verified(self):
        with patch('rag.ollama_request', return_value={'message': {'content': 'A claim [1].'}}):
            result = generate_answer('coupon', self.sources)
        self.assertEqual(result['status'], 'generated')
        self.assertTrue(result['citation_check']['valid'])

    def test_explicit_abstention_accepted(self):
        text = 'The supplied excerpts do not answer that question.'
        with patch('rag.ollama_request', return_value={'message': {'content': text}}):
            self.assertEqual(generate_answer('unknown', self.sources)['status'], 'abstained')


class RepositoryTests(unittest.TestCase):
    def test_only_repository_root_urls_accepted(self):
        self.assertEqual(validate_repository_url(' https://github.com/owner/repo.git/ '), 'https://github.com/owner/repo')
        for url in ['http://github.com/a/b', 'https://github.com.evil/a/b', 'https://github.com/a/b/tree/main',
                    'https://github.com/a/b?token=x', 'https://user@github.com/a/b', 'https://github.com/../repo']:
            with self.assertRaises(ValueError):
                validate_repository_url(url)


if __name__ == '__main__':
    unittest.main()
