import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from streamlit.testing.v1 import AppTest
from core.model_catalog import (catalog_endpoint, catalog_identity, fetch_catalog,
                                group_catalog, sync_catalog, CACHE_SECONDS)
from core.model_verifier import verify_llm_model


def model(name, capabilities, **extra):
    return dict(model=name, capabilities=capabilities, **extra)


def response(models, total=None, status=200):
    result = Mock(status_code=status)
    result.json.return_value = {'success': True, 'output': {'total': len(models) if total is None else total, 'models': models}}
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    return result


class CatalogTests(unittest.TestCase):
    def test_endpoint_keeps_region_and_workspace(self):
        for host in ['dashscope.aliyuncs.com', 'dashscope-intl.aliyuncs.com', 'abc.cn-beijing.maas.aliyuncs.com']:
            self.assertEqual(catalog_endpoint(f'https://{host}/compatible-mode/v1'), f'https://{host}/api/v1/models')
        for url in ['http://dashscope.aliyuncs.com', 'https://example.com', 'https://dashscope.aliyuncs.com.evil.test',
                    'https://user:password@dashscope.aliyuncs.com', 'https://token-plan.cn-beijing.maas.aliyuncs.com']:
            with self.assertRaises(ValueError):
                catalog_endpoint(url)

    def test_all_pages_are_loaded_and_partial_or_repeated_pages_fail(self):
        a, b = model('qwen-a', ['TG']), model('qwen-b', ['TG'])
        with patch('core.model_catalog.requests.get', side_effect=[response([a], 2), response([b], 2)]) as get:
            self.assertEqual(len(fetch_catalog('secret', 'https://dashscope.aliyuncs.com/api/v1/models')), 2)
            self.assertEqual([call.kwargs['params']['page_no'] for call in get.call_args_list], [1, 2])
        for second in [response([], 2), response([a], 2), response([b], 3), response([], status=403)]:
            with patch('core.model_catalog.requests.get', side_effect=[response([a], 2), second]):
                with self.assertRaises(ValueError):
                    fetch_catalog('secret', 'https://dashscope.aliyuncs.com/api/v1/models')

    def test_classification_offline_and_api_compatibility(self):
        groups, notices = group_catalog([
            model('qwen-plus', ['TG']), model('deepseek-new', ['TG', 'Reasoning']),
            model('qwen3-vl-plus', ['VU']), model('qwen3-asr-flash-2026-10-01', ['ASR']),
            model('qwen3-asr-flash-realtime', ['Realtime-ASR']), model('paraformer-v2', ['ASR']),
            model('qwen-image', ['IG']), model('text-embedding-v4', ['TR']),
            model('other-provider', ['TG'], inference_provider='siliconflow'),
            model('retired', ['TG'], inference_offline_info={'offline_time': '2020-01-01'}),
            model('retiring', ['TG'], inference_offline_info={'offline_time': '2099-01-01'}),
        ], now=datetime(2026, 11, 6, tzinfo=timezone.utc))
        self.assertEqual(groups['llm_models'], ['deepseek-new', 'qwen-plus', 'retiring'])
        self.assertEqual(groups['asr_models'], ['qwen3-asr-flash-2026-10-01'])
        self.assertEqual(groups['vlm_models'], ['qwen3-vl-plus'])
        self.assertIn('2099', notices['retiring'])

    def test_cache_refresh_failure_preserves_last_complete_catalog_and_no_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            state = {'llm_models': ['old'], 'asr_models': ['local-asr'], 'vlm_models': ['local-vlm']}
            url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
            secret = 'sk-test-secret'
            with patch('core.model_catalog.fetch_catalog', return_value=[model('new', ['TG'])]) as fetch:
                sync_catalog(state, secret, url, now=100000, cache_dir=directory)
                self.assertEqual(state['llm_models'], ['new'])
                sync_catalog(state, secret, url, now=100500, cache_dir=directory)
                self.assertEqual(fetch.call_count, 1)
                sync_catalog(state, secret, url, force=True, now=100501, cache_dir=directory)
                self.assertEqual(fetch.call_count, 2)
            cached_path = next(Path(directory).glob('*.json'))
            original = cached_path.read_text(encoding='utf-8')
            self.assertNotIn(secret, original)
            state['llm_models'].append('manual-model')
            with patch('core.model_catalog.fetch_catalog', side_effect=requests.Timeout):
                sync_catalog(state, secret, url, force=True, now=100501 + CACHE_SECONDS, cache_dir=directory)
            self.assertEqual(state['llm_models'], ['new', 'manual-model'])
            self.assertEqual(cached_path.read_text(encoding='utf-8'), original)
            self.assertIn('同步失败', state['catalog_message'])
            self.assertEqual(state['asr_models'], ['local-asr'])

    def test_account_and_region_cache_isolation_and_auto_retry_throttle(self):
        with tempfile.TemporaryDirectory() as directory:
            state = {'catalog_local_models': {'llm_models': ['local']}}
            url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
            with patch('core.model_catalog.fetch_catalog', return_value=[model('account-a', ['TG'])]):
                sync_catalog(state, 'key-a', url, now=100000, cache_dir=directory)
            with patch('core.model_catalog.fetch_catalog', side_effect=requests.Timeout) as fetch:
                sync_catalog(state, 'key-b', url, now=100001, cache_dir=directory)
                sync_catalog(state, 'key-b', url, now=100002, cache_dir=directory)
                self.assertEqual(fetch.call_count, 1)
            self.assertEqual(state['llm_models'], ['local'])
            self.assertIsNone(state['catalog_synced_at'])
            self.assertNotEqual(catalog_identity('key', 'region-a'), catalog_identity('key', 'region-b'))

    def test_invalid_catalog_does_not_replace_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            state = {}
            url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
            with patch('core.model_catalog.fetch_catalog', return_value=[model('good', ['TG'])]):
                sync_catalog(state, 'key', url, now=100000, cache_dir=directory)
            with patch('core.model_catalog.fetch_catalog', return_value=[model('image-only', ['IG'])]):
                sync_catalog(state, 'key', url, force=True, now=100001, cache_dir=directory)
            self.assertEqual(state['llm_models'], ['good'])
            self.assertEqual(json.loads(next(Path(directory).glob('*.json')).read_text())['fetched_at'], 100000)

    def test_quota_error_is_explained_without_claiming_key_invalid(self):
        failed = Mock(status_code=403)
        failed.json.return_value = {'error': {'message': 'Free quota exhausted. use free tier only'}}
        with patch('core.model_verifier.requests.post', return_value=failed):
            success, message = verify_llm_model('key', 'https://example.invalid/v1', 'model')
        self.assertFalse(success)
        self.assertIn('免费额度已用完', message)
        self.assertNotIn('未知错误', message)

    def test_verification_page_syncs_without_generation_calls_and_resets_stale_selection(self):
        def sync(state, *args, **kwargs):
            state['llm_models'] = ['qwen-new']
            state['catalog_synced_at'] = 100000
            state['catalog_notices'] = {}
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'))
        app.session_state['runtime_config'] = {'api_key': 'sk-test', 'base_url': 'https://dashscope.aliyuncs.com/compatible-mode/v1'}
        app.session_state['verify_model_select'] = 'retired-model'
        with patch('core.model_catalog.sync_catalog', side_effect=sync) as sync_call, \
             patch('core.model_verifier.verify_llm_model', return_value=(False, '免费额度已用完')) as verify:
            app.run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.selectbox(key='verify_model_select').value, 'qwen-new')
            verify.assert_not_called()
            app.button(key='refresh_catalog_verify').click().run()
            self.assertTrue(sync_call.call_args.kwargs['force'])
            next(button for button in app.button if button.label == '验证并保存').click().run()
            verify.assert_called_once_with('sk-test', 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'qwen-new')
            self.assertEqual(len(app.exception), 0)


if __name__ == '__main__':
    unittest.main()
