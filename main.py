import concurrent.futures
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path

from core.config import LLM_CONFIG
from core.doc_parser import parse_reference_files, parse_document, SUPPORTED_DOCUMENT_EXTS
from core.note_assets import create_assets_dir
from core.tasks import (TaskCancelled, check_cancelled, completed_futures,
                        create_task_paths, snapshot_config)
from media_processor.splitter import split_media_to_audio_chunks_generator
from media_processor.transcriber import transcribe_with_qwen
from media_processor.visual_manager import VisualManager
from ai_services.llm_api import run_llm_generation

VIDEO_EXTS = {'.mp4', '.mov', '.mpeg', '.webm'}
AUDIO_EXTS = {'.mp3', '.m4a', '.wav', '.amr', '.mpga'}
TEXT_EXTS = SUPPORTED_DOCUMENT_EXTS


def main_process_generator(input_path, output_filename, stream_output, note_type,
                           asr_context=None, additional_instructions='',
                           qwen_asr_model='qwen3-asr-flash', enable_visual_analysis=False,
                           vlm_model_name='qwen3-vl-plus', vlm_backup_model=None,
                           keep_visual_files=False, insert_images=False, vlm_concurrency=10,
                           asr_concurrency=10, reference_files=None, transcript_only=False,
                           keep_transcript=False, config=None, cancel_event=None):
    # 在调用时冻结配置，不能等后台开始执行后再读取可变的全局/会话配置。
    settings = snapshot_config(LLM_CONFIG if config is None else config)
    return _run(input_path, output_filename, stream_output, note_type, asr_context,
                additional_instructions, qwen_asr_model, enable_visual_analysis,
                vlm_model_name, vlm_backup_model, keep_visual_files, insert_images,
                vlm_concurrency, asr_concurrency, reference_files, transcript_only,
                keep_transcript, settings, cancel_event or threading.Event())


def _run(input_path, output_filename, stream_output, note_type, asr_context,
         additional_instructions, qwen_asr_model, enable_visual_analysis,
         vlm_model_name, vlm_backup_model, keep_visual_files, insert_images,
         vlm_concurrency, asr_concurrency, reference_files, transcript_only,
         keep_transcript, config, cancel_event):
    task_id = None
    work = None
    logs = []
    status = 'failed'

    def log(message):
        logs.append(f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}')

    def save_text(path, content):
        check_cancelled(cancel_event)
        path.write_text(content, encoding='utf-8')

    try:
        check_cancelled(cancel_event)
        task_id, work, output = create_task_paths(output_filename)
        output_filename = output_filename.strip()
        yield 'output_dir', str(output)
        log(f'任务开始: {input_path}; ASR={qwen_asr_model}; LLM={config.get("model")}')
        reference_content = parse_reference_files(reference_files) if reference_files else ''
        suffix = Path(input_path).suffix.lower()
        if suffix in TEXT_EXTS:
            yield 'progress', 0.1, '正在解析文档...'
            full_transcript = parse_document(input_path)
        elif suffix in VIDEO_EXTS | AUDIO_EXTS:
            chunk_dir = work / 'audio'
            yield 'progress', 0.05, '正在切分音频...'
            chunks = []
            for kind, value, *extra in split_media_to_audio_chunks_generator(
                    input_path, str(chunk_dir), 170, cancel_event=cancel_event):
                check_cancelled(cancel_event)
                if kind == 'error':
                    raise RuntimeError(value)
                if kind == 'result':
                    chunks = value
                elif kind == 'progress':
                    yield 'sub_progress', value / extra[0], f'正在切分... ({value}/{extra[0]})'
            if not chunks:
                raise RuntimeError('未能提取出音频块')
            yield 'progress', 0.25, '正在转录音频...'
            transcripts = [None] * len(chunks)
            with concurrent.futures.ThreadPoolExecutor(max_workers=asr_concurrency) as executor:
                futures = {
                    executor.submit(transcribe_with_qwen, chunk, asr_context, qwen_asr_model,
                                    start_offset_seconds=i * 170, config=config,
                                    cancel_event=cancel_event): i
                    for i, chunk in enumerate(chunks)
                }
                try:
                    for count, future in enumerate(completed_futures(futures, cancel_event), 1):
                        result = future.result()
                        if result is None:
                            raise RuntimeError(f'音频块 {futures[future] + 1} 转录失败')
                        transcripts[futures[future]] = result
                        yield 'sub_progress', count / len(chunks), f'正在转录... ({count}/{len(chunks)})'
                finally:
                    for future in futures:
                        future.cancel()
            check_cancelled(cancel_event)
            full_transcript = '\n\n'.join(filter(None, transcripts))
            if not full_transcript.strip():
                raise ValueError('音频未识别出有效文字')
            shutil.rmtree(chunk_dir, ignore_errors=True)
            if suffix in VIDEO_EXTS and enable_visual_analysis and not transcript_only:
                yield 'progress', 0.55, '正在分析视频画面...'
                assets = create_assets_dir(str(output)) if insert_images else None
                yield 'assets_dir', assets
                frame_dir = output / 'frames' if keep_visual_files else work / 'frames'
                visual_report = ''
                try:
                    for kind, value, *extra in VisualManager().process_video_generator(
                            input_path, interval_seconds=60, mode='lecture_mixed',
                            max_workers=vlm_concurrency, model=vlm_model_name,
                            backup_model=vlm_backup_model, keep_intermediate_files=keep_visual_files,
                            insert_images=insert_images, image_output_dir=assets,
                            frame_dir=str(frame_dir), config=config, cancel_event=cancel_event):
                        check_cancelled(cancel_event)
                        if kind == 'stage_progress':
                            current, total = extra
                            yield 'sub_progress', current / total if total else 0, f'{value}: {current}/{total}'
                        elif kind == 'log':
                            message = extra[0] if extra else value
                            log(f'[VisualManager] {message}')
                            yield 'sub_progress', 0.0, str(message)
                        elif kind == 'result':
                            visual_report = value
                    if keep_visual_files:
                        save_text(output / f'{output_filename}_visual_report.md', visual_report)
                    full_transcript += '\n\n【视频视觉分析】\n' + visual_report
                    if insert_images:
                        full_transcript += '\n请将上述 Markdown 图片链接原样插入笔记相应段落。'
                except TaskCancelled:
                    raise
                except Exception as error:
                    log(f'视觉分析失败: {error}')
                    yield 'sub_progress', 1.0, f'视觉分析失败，使用音频继续：{error}'
        else:
            raise ValueError(f'不支持的文件类型: {suffix}')

        check_cancelled(cancel_event)
        yield 'transcript', full_transcript
        if keep_transcript or transcript_only:
            transcript_path = output / f'{output_filename}.txt'
            save_text(transcript_path, full_transcript)
            yield 'transcript_saved', str(transcript_path)
        if transcript_only:
            final_path = transcript_path
        else:
            yield 'progress', 0.8, '正在生成笔记...'
            result = run_llm_generation(
                full_transcript, stream_output, note_type, additional_instructions,
                reference_material=reference_content, config=config, cancel_event=cancel_event)
            final_text = ''
            if stream_output:
                try:
                    for text in result:
                        check_cancelled(cancel_event)
                        if text:
                            final_text += text
                            yield 'llm_chunk', text
                finally:
                    close = getattr(result, 'close', None)
                    if close:
                        close()
            else:
                check_cancelled(cancel_event)
                final_text = result
                yield 'llm_chunk', final_text
            if not final_text:
                raise ValueError('模型未返回有效内容')
            final_path = output / f'{output_filename}.md'
            save_text(final_path, final_text)
        status = 'success'
        log(f'完成: {final_path}')
        yield 'progress', 1.0, '处理完成'
        yield 'done', str(final_path), '生成完成！'
    except (TaskCancelled, GeneratorExit):
        if status != 'success':
            status = 'stopped'
        cancel_event.set()
        log('用户停止任务')
        raise
    except Exception as error:
        log(f'处理失败: {error}')
        yield 'persistent_error', 0, str(error)
    finally:
        # 只清理本任务；线程池退出后已没有后台线程使用这些文件。
        if work is not None:
            shutil.rmtree(work, ignore_errors=True)
        if task_id:
            Path('logs').mkdir(exist_ok=True)
            Path('logs', f'{task_id}_{status}.txt').write_text('\n'.join(logs), encoding='utf-8')
