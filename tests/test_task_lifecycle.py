import concurrent.futures
import os
import sys
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main
from core.tasks import (TaskCancelled, background_events, run_process,
                        validate_output_name)
from core.utils import APIError, retry, retryable
from media_processor.transcriber import transcribe_with_qwen


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.previous = os.getcwd()
        os.chdir(self.directory.name)

    def tearDown(self):
        os.chdir(self.previous)
        self.directory.cleanup()

    def test_concurrent_same_name_tasks_use_frozen_configs_and_separate_files(self):
        Path('source.txt').write_text('transcript', encoding='utf-8')
        barrier = threading.Barrier(2)
        def generate(*args, config, **kwargs):
            barrier.wait(timeout=5)
            return iter([config['model']])
        settings = {'model': 'first', 'api_key': 'first-key'}
        first = main.main_process_generator('source.txt', 'same', True, 'STEM',
                                            config=settings, keep_transcript=True)
        settings['model'] = 'second'
        second = main.main_process_generator('source.txt', 'same', True, 'STEM',
                                             config=settings, keep_transcript=True)
        settings['model'] = 'changed after launch'
        with patch('main.run_llm_generation', side_effect=generate), concurrent.futures.ThreadPoolExecutor(2) as pool:
            a, b = pool.submit(list, first), pool.submit(list, second)
            results = [a.result(timeout=10), b.result(timeout=10)]
        paths = [Path(next(e[1] for e in events if e[0] == 'done')) for events in results]
        self.assertNotEqual(paths[0], paths[1])
        self.assertEqual([p.read_text(encoding='utf-8') for p in paths], ['first', 'second'])
        for path in paths:
            self.assertEqual(path.with_suffix('.txt').read_text(encoding='utf-8'), 'transcript')
        self.assertEqual(list(Path('temp/tasks').iterdir()), [])
        self.assertEqual(len(list(Path('logs').glob('*_success.txt'))), 2)

    def test_transcript_survives_later_task_and_generation_error(self):
        Path('source.txt').write_text('keep this', encoding='utf-8')
        with patch('main.run_llm_generation', side_effect=RuntimeError('model failed')):
            events = list(main.main_process_generator('source.txt', 'notes', True, 'STEM', keep_transcript=True))
        saved = Path(next(e[1] for e in events if e[0] == 'transcript_saved'))
        self.assertTrue(any(e[0] == 'persistent_error' for e in events))
        with patch('main.run_llm_generation', return_value=iter(['new notes'])):
            list(main.main_process_generator('source.txt', 'notes', True, 'STEM'))
        self.assertEqual(saved.read_text(encoding='utf-8'), 'keep this')
        self.assertEqual(len(list(Path('output').rglob('*.txt'))), 1)

    def test_invalid_names_fail_before_processing(self):
        for name in ('', '../escape', 'a/b', 'a\\b', 'C:escape', 'CON', 'NUL.txt', 'bad?', 'bad.', 'x' * 65):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_output_name(name)
                events = list(main.main_process_generator('unused', name, True, 'STEM'))
                self.assertEqual(events[0][0], 'persistent_error')
        self.assertFalse(Path('output').exists())
        self.assertEqual(validate_output_name('课程笔记'), '课程笔记')

    def test_background_cancel_returns_without_waiting_for_network(self):
        started, release, cleaned, cancel = (threading.Event() for _ in range(4))
        def events():
            started.set()
            release.wait(3)
            yield 'done', 'late result'
        bridge = background_events(events(), cancel, cleanup=cleaned.set)
        self.assertEqual(next(bridge)[0], 'heartbeat')
        self.assertTrue(started.is_set())
        cancel.set()
        before = time.monotonic()
        with self.assertRaises(TaskCancelled):
            next(bridge)
        self.assertLess(time.monotonic() - before, 0.5)
        release.set()
        self.assertTrue(cleaned.wait(3))

    def test_cancellation_during_llm_does_not_save_notes(self):
        Path('source.txt').write_text('source', encoding='utf-8')
        cancel = threading.Event()
        def generate(*args, **kwargs):
            yield 'partial'
            cancel.set()
            yield 'late'
        with patch('main.run_llm_generation', side_effect=generate):
            with self.assertRaises(TaskCancelled):
                list(main.main_process_generator('source.txt', 'notes', True, 'STEM', cancel_event=cancel))
        self.assertEqual(list(Path('output').rglob('*.md')), [])
        self.assertEqual(list(Path('temp/tasks').iterdir()), [])

    def test_media_process_is_terminated_on_cancel(self):
        cancel = threading.Event()
        timer = threading.Timer(0.2, cancel.set)
        timer.start()
        started = time.monotonic()
        try:
            with self.assertRaises(TaskCancelled):
                run_process([sys.executable, '-c', 'import time; time.sleep(20)'], cancel)
        finally:
            timer.cancel()
        self.assertLess(time.monotonic() - started, 3)

    def test_real_ffmpeg_audio_pipeline(self):
        with wave.open('audio.wav', 'wb') as audio:
            audio.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            audio.writeframes(b'\x00\x00' * 4000)
        def transcribe(path, *args, **kwargs):
            self.assertTrue(Path(path).is_file())
            self.assertGreater(Path(path).stat().st_size, 0)
            return '[00:00] fixture transcript'
        with patch('main.transcribe_with_qwen', side_effect=transcribe):
            events = list(main.main_process_generator('audio.wav', 'audio', True, 'STEM', transcript_only=True))
        path = Path(next(e[1] for e in events if e[0] == 'done'))
        self.assertEqual(path.read_text(encoding='utf-8'), '[00:00] fixture transcript')
        self.assertEqual(list(Path('temp/tasks').iterdir()), [])


class RetryTests(unittest.TestCase):
    def test_retry_classification_and_limit(self):
        for status, code, expected in [(401, 'InvalidApiKey', False), (400, 'BadRequest', False),
                                      (429, 'Throttling.RateQuota', True),
                                      (429, 'AllocationQuota.Exhausted', False), (503, 'Unavailable', True)]:
            self.assertEqual(retryable(APIError(status, code, 'error')), expected)
        calls = Mock(side_effect=APIError(503, 'Unavailable', 'retry'))
        with self.assertRaises(APIError):
            retry(max_retries=3, delay=0)(calls)()
        self.assertEqual(calls.call_count, 3)
        calls = Mock(side_effect=APIError(401, 'InvalidApiKey', 'stop'))
        with self.assertRaises(APIError):
            retry(max_retries=3, delay=0)(calls)()
        self.assertEqual(calls.call_count, 1)

    def test_cancel_interrupts_retry_wait(self):
        cancel = threading.Event()
        calls = Mock(side_effect=APIError(503, 'Unavailable', 'retry'))
        timer = threading.Timer(0.1, cancel.set)
        timer.start()
        before = time.monotonic()
        try:
            with self.assertRaises(TaskCancelled):
                retry(delay=30)(calls)(cancel_event=cancel)
        finally:
            timer.cancel()
        self.assertEqual(calls.call_count, 1)
        self.assertLess(time.monotonic() - before, 2)

    def test_missing_file_and_invalid_key_do_not_retry_or_use_backup(self):
        with patch('media_processor.transcriber.dashscope.MultiModalConversation.call') as call:
            with self.assertRaises(FileNotFoundError):
                transcribe_with_qwen('nonexistent.mp3', config={'api_key': 'test'})
            call.assert_not_called()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.mp3'
            source.write_bytes(b'audio')
            with patch('media_processor.transcriber.dashscope.MultiModalConversation.call',
                       return_value=SimpleNamespace(status_code=401, code='InvalidApiKey', message='invalid')) as call:
                with self.assertRaises(APIError):
                    transcribe_with_qwen(str(source), config={'api_key': 'test', 'asr_backup_model': 'backup'})
                self.assertEqual(call.call_count, 1)
                self.assertEqual(call.call_args.kwargs['api_key'], 'test')


if __name__ == '__main__':
    unittest.main()
