import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from core.config import LLM_CONFIG
from core.tasks import snapshot_config


class AppStateTests(unittest.TestCase):
    def test_two_sessions_render_without_mutating_shared_config(self):
        initial = snapshot_config(LLM_CONFIG)
        app_path = Path(__file__).resolve().parents[1] / 'app.py'
        apps = []
        with patch('requests.post', side_effect=AssertionError('Unexpected network request')), \
             patch('dashscope.MultiModalConversation.call', side_effect=AssertionError('Unexpected model request')), \
             patch('core.config.save_environment') as save:
            for key in ('session-one', 'session-two'):
                app = AppTest.from_file(str(app_path), default_timeout=10)
                app.session_state['runtime_config'] = {
                    'api_key': key, 'asr_api_key': key, 'model': 'qwen-plus',
                    'base_url': 'https://example.invalid/v1', 'provider_name': 'test',
                }
                app.session_state['has_validated_on_startup'] = True
                app.run()
                self.assertEqual(len(app.exception), 0)
                apps.append(app)
            apps[0].session_state['runtime_config']['api_key'] = 'changed'
            self.assertEqual(apps[1].session_state['runtime_config']['api_key'], 'session-two')
            filename_input = next(item for item in apps[0].text_input if '笔记文件名' in item.label)
            filename_input.set_value('../invalid').run()
            self.assertEqual(len(apps[0].exception), 0)
            self.assertTrue(any('文件名' in item.value for item in apps[0].error))
            save.assert_not_called()
        self.assertEqual(LLM_CONFIG, initial)


if __name__ == '__main__':
    unittest.main()
