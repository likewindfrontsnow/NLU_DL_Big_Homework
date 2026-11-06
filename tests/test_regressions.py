import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from docx import Document
from pptx import Presentation
from pptx.util import Inches
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

import main
from ai_services.llm_api import _handle_streaming_response, run_llm_generation
from core.doc_parser import parse_document
from core.note_assets import create_assets_dir, asset_relative_path, add_assets_to_zip
from core.uploads import reset_upload_state


class FakeResponse:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def iter_lines(self):
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            payload = event if isinstance(event, str) else json.dumps(event)
            yield ('data: ' + payload).encode('utf-8')

    def close(self):
        self.closed = True


def chunk(text):
    return {'choices': [{'delta': {'content': text}}]}


class StreamTests(unittest.TestCase):
    def test_success_with_done_or_stop_and_usage(self):
        for ending in (
            ['[DONE]'],
            [{'choices': [{'delta': {}, 'finish_reason': 'stop'}]}, {'choices': [], 'usage': {}}],
        ):
            response = FakeResponse([chunk('complete')] + ending)
            self.assertEqual(list(_handle_streaming_response(response)), ['complete'])
            self.assertTrue(response.closed)

    def test_errors_truncation_and_malformed_stream_are_rejected(self):
        endings = [
            [],
            [{'error': {'message': 'quota exhausted'}}, '[DONE]'],
            [{'choices': [{'delta': {}, 'finish_reason': 'length'}]}, '[DONE]'],
            [{'choices': [{'delta': {}, 'finish_reason': 'content_filter'}]}],
            ['invalid json'],
            [ConnectionError('connection lost')],
        ]
        for ending in endings:
            with self.subTest(ending=ending):
                response = FakeResponse([chunk('partial')] + ending)
                with self.assertRaises((RuntimeError, ValueError, ConnectionError)):
                    list(_handle_streaming_response(response))
                self.assertTrue(response.closed)

    def test_custom_instructions_preserve_braces_for_all_note_types(self):
        instruction = r'Use \frac{a}{b} and {"title": "notes"}'
        for kind in ('STEM', 'HASS', 'Medical', 'None'):
            with self.subTest(kind=kind), patch('ai_services.llm_api._call_llm_api') as call:
                run_llm_generation('source {text}', False, kind, instruction, 'reference {text}')
                prompt = call.call_args.args[0][0]['content']
                self.assertIn(instruction, prompt)
                self.assertIn('source {text}', prompt)
                self.assertIn('reference {text}', prompt)


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.previous_directory = os.getcwd()
        os.chdir(self.directory.name)

    def tearDown(self):
        os.chdir(self.previous_directory)
        self.directory.cleanup()

    def run_process(self, path, **kwargs):
        return list(main.main_process_generator(
            str(path), 'notes', True, 'STEM', **kwargs
        ))

    def create_documents(self):
        doc = Document()
        doc.add_paragraph('Document fixture')
        doc.save('sample.docx')
        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])
        slide.shapes.add_textbox(Inches(1), Inches(1), Inches(5), Inches(1)).text = 'Slide fixture'
        presentation.save('sample.pptx')
        writer = PdfWriter()
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                                 NameObject('/Subtype'): NameObject('/Type1'),
                                 NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({
            NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})
        })
        content = DecodedStreamObject()
        content.set_data(b'BT /F1 12 Tf 20 200 Td (PDF fixture) Tj ET')
        page[NameObject('/Contents')] = writer._add_object(content)
        writer.write('sample.pdf')
        Path('sample.txt').write_bytes('中文文本'.encode('gbk'))
        return {'sample.docx': 'Document fixture', 'sample.pptx': 'Slide fixture',
                'sample.pdf': 'PDF fixture', 'sample.txt': '中文文本'}

    def test_real_documents_reach_generation_as_extracted_text(self):
        for filename, expected in self.create_documents().items():
            with self.subTest(filename=filename), patch('main.run_llm_generation', return_value=iter(['notes'])) as llm:
                events = self.run_process(filename)
                self.assertIn(expected, llm.call_args.args[0])
                self.assertTrue(any(event[0] == 'done' for event in events))

    def test_bad_empty_and_unsupported_documents_fail_before_model_call(self):
        for filename, content in [('bad.pdf', b'broken'), ('empty.txt', b''), ('legacy.doc', b'binary')]:
            Path(filename).write_bytes(content)
            with self.subTest(filename=filename), patch('main.run_llm_generation') as llm:
                events = self.run_process(filename)
                self.assertTrue(any(event[0] == 'persistent_error' for event in events))
                self.assertFalse(any(event[0] == 'done' for event in events))
                llm.assert_not_called()

    def test_stream_error_does_not_save_or_report_success(self):
        Path('input.txt').write_text('source', encoding='utf-8')
        response = FakeResponse([chunk('partial'), {'error': {'message': 'failed'}}, '[DONE]'])
        with patch('main.run_llm_generation', return_value=_handle_streaming_response(response)):
            events = self.run_process('input.txt')
        self.assertTrue(any(event[0] == 'persistent_error' for event in events))
        self.assertFalse(any(event[0] == 'done' for event in events))
        self.assertEqual(list(Path('output').rglob('*.md')), [])

    def test_chunk_offsets_for_cloud_transcription(self):
        Path('input.mp4').write_bytes(b'fixture')
        with patch('main.split_media_to_audio_chunks_generator', return_value=iter([('result', ['one', 'two', 'three'])])), \
             patch('main.transcribe_with_qwen', side_effect=lambda *args, **kw: str(kw['start_offset_seconds'])) as transcribe:
            events = self.run_process('input.mp4', transcript_only=True)
            offsets = sorted(call.kwargs['start_offset_seconds'] for call in transcribe.call_args_list)
            self.assertEqual(offsets, [0, 170, 340])
            transcript = next(event[1] for event in events if event[0] == 'transcript')
            self.assertEqual(transcript, '0\n\n170\n\n340')

    def test_image_archives_are_isolated_and_links_resolve(self):
        first = create_assets_dir('output')
        second = create_assets_dir('output')
        Path(first, 'frame.jpg').write_bytes(b'first video')
        Path(second, 'frame.jpg').write_bytes(b'second video')
        Path('output/assets/legacy.jpg').write_bytes(b'legacy')
        self.assertNotEqual(first, second)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            link = asset_relative_path(second, 'frame.jpg')
            archive.writestr('notes.md', f'![frame]({link})')
            add_assets_to_zip(archive, second)
        with zipfile.ZipFile(buffer) as archive:
            self.assertEqual(set(archive.namelist()), {'notes.md', link})
            self.assertEqual(archive.read(link), b'second video')
        self.assertEqual(Path(first, 'frame.jpg').read_bytes(), b'first video')

    def test_consecutive_video_runs_keep_their_own_visual_assets(self):
        Path('input.mp4').write_bytes(b'fixture')
        def split(*args, **kwargs):
            Path(args[1]).mkdir(parents=True, exist_ok=True)
            yield 'result', ['audio']

        saved_images = []
        for image_content in (b'first video', b'second video'):
            def extract(*args, **kwargs):
                frame = Path(kwargs['output_dir']) / 'frame_001.jpg'
                frame.parent.mkdir(parents=True, exist_ok=True)
                frame.write_bytes(image_content)
                yield 'result', [str(frame)]

            with patch('main.split_media_to_audio_chunks_generator', side_effect=split), \
                 patch('main.transcribe_with_qwen', return_value='audio transcript'), \
                 patch('media_processor.frame_extractor.FrameExtractor.extract_frames_generator', side_effect=extract), \
                 patch('media_processor.visual_manager.analyze_image', return_value='visual text'), \
                 patch('main.run_llm_generation', side_effect=lambda text, *args, **kwargs: iter([text])):
                events = self.run_process('input.mp4', enable_visual_analysis=True, insert_images=True)

            directory = next(event[1] for event in events if event[0] == 'assets_dir')
            self.assertTrue(any(e[0] == 'sub_progress' and '图像去重' in str(e[2]) for e in events))
            image = next(Path(directory).iterdir())
            link = asset_relative_path(directory, image.name)
            notes = Path(next(event[1] for event in events if event[0] == 'done')).read_text(encoding='utf-8')
            self.assertIn(f']({link})', notes)
            self.assertEqual((Path(directory).parent.parent / link).read_bytes(), image_content)
            self.assertTrue(any(event[0] == 'done' for event in events))
            saved_images.append(image)
        self.assertNotEqual(saved_images[0], saved_images[1])
        self.assertEqual(saved_images[0].read_bytes(), b'first video')
        self.assertEqual(saved_images[1].read_bytes(), b'second video')


class UploadStateTests(unittest.TestCase):
    def test_same_name_replacement_and_removal_clear_stale_results(self):
        for upload in (io.BytesIO(b'new content'), None):
            state = {'source_upload': upload, 'last_uploaded_filename': 'lecture.mp4',
                     'current_notes': 'old notes', 'full_transcript': 'old transcript',
                     'current_assets_dir': 'old assets', 'processing_has_failed': True,
                     'processing_started': True, 'output_filename': 'my notes'}
            reset_upload_state(state)
            self.assertIsNone(state['current_notes'])
            self.assertIsNone(state['full_transcript'])
            self.assertIsNone(state['current_assets_dir'])
            self.assertFalse(state['processing_has_failed'])
            self.assertFalse(state['processing_started'])
            self.assertEqual(state['output_filename'], 'my notes')
            self.assertIs(state['source_upload'], upload)


if __name__ == '__main__':
    unittest.main()
