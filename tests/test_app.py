import unittest
from unittest.mock import patch
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


class InterfaceTests(unittest.TestCase):
    def test_fresh_start_sample_search_and_clear(self):
        with patch('rag.ollama_request', side_effect=AssertionError('Source mode must work offline')):
            app = AppTest.from_file(APP, default_timeout=15).run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn('Demo Shop', [metric.value for metric in app.metric])
            app.chat_input[0].set_value('When is domestic shipping free?').run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.session_state['messages']), 2)
            self.assertEqual(app.session_state['messages'][1]['sources'][0]['file_path'], 'shipping.py')
            clear = next(button for button in app.button if button.label == 'Clear conversation')
            clear.click().run()
            self.assertEqual(len(app.session_state['messages']), 0)
            self.assertEqual(app.session_state['index']['repository']['name'], 'Demo Shop')

    def test_invalid_repository_retains_snapshot(self):
        app = AppTest.from_file(APP, default_timeout=15).run()
        app.text_input[0].set_value('https://example.com/not-a-repo')
        button = next(button for button in app.button if button.label == 'Load repository')
        button.click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.error)
        self.assertEqual(app.session_state['index']['repository']['name'], 'Demo Shop')

    def test_missing_ollama_explained_without_crash(self):
        with patch('rag.available_models', side_effect=RuntimeError('Ollama unavailable')):
            app = AppTest.from_file(APP, default_timeout=15).run()
            app.radio[0].set_value('Local AI').run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(app.warning)
            self.assertTrue(app.chat_input[0].disabled)


if __name__ == '__main__':
    unittest.main()
