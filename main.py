import concurrent.futures
import time
import os
import shutil
from datetime import datetime
from core.config import LLM_CONFIG
from media_processor.splitter import split_media_to_audio_chunks_generator
from media_processor.transcriber import transcribe_single_audio_chunk, pre_download_whisper_model, transcribe_with_qwen
from media_processor.visual_manager import VisualManager
from ai_services.llm_api import run_llm_generation
from core.doc_parser import parse_reference_files

VIDEO_EXTS = {'.mp4', '.mov', '.mpeg', '.webm'}
AUDIO_EXTS = {'.mp3', '.m4a', '.wav', '.amr', '.mpga'}
TEXT_EXTS = {'.txt', '.md', '.mdx', '.markdown', '.pdf', '.html', '.xlsx', '.xls', '.doc', '.docx', '.csv', '.eml', '.msg', '.pptx', '.ppt', '.xml', '.epub'}

def main_process_generator(
    input_path: str,
    output_filename: str,
    whisper_model_size: str,
    stream_output: bool,
    transcription_provider: str,
    note_type: str,
    asr_context: str | None = None,
    additional_instructions: str = "",
    qwen_asr_model: str = "qwen3-asr-flash",
    enable_visual_analysis: bool = False,
    vlm_model_name: str = "qwen3-vl-plus",
    vlm_backup_model: str = None,
    keep_visual_files: bool = False,
    insert_images: bool = False,
    vlm_concurrency: int = 10,
    asr_concurrency: int = 10,
    reference_files: list = None,
    transcript_only: bool = False
): 
    run_logs = []
    task_completed = False

    def log(msg):
        run_logs.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

    def save_run_log(status="completed"):
        try:
            os.makedirs("logs", exist_ok=True)
            log_name = f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{status}.txt"
            with open(os.path.join("logs", log_name), "w", encoding="utf-8") as f:
                f.write("\n".join(run_logs))
            return log_name
        except Exception:
            return None

    log(f"任务开始: {input_path}")
    log(f"参数: ASR={transcription_provider} (Workers: {asr_concurrency}), LLM={LLM_CONFIG.get('model')}, Note={note_type}")
    log(f"视觉: {enable_visual_analysis} (Model: {vlm_model_name}, Backup: {vlm_backup_model}, Workers: {vlm_concurrency})")
    if reference_files:
        log(f"参考资料数量: {len(reference_files)}")

    temp_root = "temp"
    output_dir = os.path.join(temp_root, "output_chunks")
    final_output_dir = "output"
    os.makedirs(final_output_dir, exist_ok=True)

    final_notes_save_path = os.path.join(final_output_dir, f"{output_filename}.md")
    
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir) 
            time.sleep(0.5)          
        except OSError:
            pass

    os.makedirs(temp_root, exist_ok=True)
    
    transcript_save_path = os.path.join(temp_root, "source_transcript.txt")
    if os.path.exists(transcript_save_path):
        try:
            os.remove(transcript_save_path)
        except OSError:
            pass
        
    file_ext = os.path.splitext(input_path)[1].lower()
    current_progress = 0
    full_transcript = ""
    reference_content = ""

    try:
        if reference_files:
            yield "progress_text", "正在解析参考资料..."
            reference_content = parse_reference_files(reference_files)
            log(f"参考资料解析完成，长度: {len(reference_content)}")

        def run_llm_and_yield_results():
            nonlocal task_completed
            final_text = "" 
            try:
                provider_name = LLM_CONFIG.get('provider_name', 'LLM')
                model_name = LLM_CONFIG.get('model', 'default')
                stream_status = "流式" if stream_output else "非流式"
                yield "progress_text", f"正在提交给 {provider_name} (模型: {model_name}, 模式: {stream_status}, 类型: {note_type})..."
                log(f"开始调用 LLM: {model_name}")
                
                llm_call_result = run_llm_generation(
                    full_transcript, 
                    stream_output, 
                    note_type, 
                    additional_instructions,
                    reference_material=reference_content
                )
                
                if stream_output:
                    for chunk in llm_call_result:
                        if chunk:
                            final_text += chunk
                            yield "llm_chunk", chunk
                else:
                    final_text = llm_call_result
                    if final_text:
                        yield "llm_chunk", final_text 
                
                if not final_text:
                    err_msg = "API 未返回有效内容"
                    log(f"LLM 错误: {err_msg}")
                    yield "persistent_error", 0, f"**笔记生成失败**\n\n{err_msg}"
                    save_run_log("failed")
                    return

                try:
                    with open(final_notes_save_path, 'w', encoding='utf-8') as f:
                        f.write(final_text)
                    log(f"笔记已保存: {final_notes_save_path}")
                    yield "save_path", final_notes_save_path
                except IOError as e:
                    log(f"文件保存失败: {e}")
                    yield "persistent_error", 0, f"**保存最终笔记文件失败**\n\n`{e}`"
                    save_run_log("failed")
                    return

            except Exception as e: 
                log(f"LLM 流程异常: {e}")
                yield "persistent_error", 0, f"**笔记生成失败**\n\n`{e}`"
                save_run_log("failed")
                return
            
        if file_ext in TEXT_EXTS:
            total_steps = 1 if transcript_only else 2
            yield "progress", 0 / total_steps, "步骤 1/1: 正在读取文本文档..." if transcript_only else "步骤 1/2: 正在读取文本文档..."
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    full_transcript = f.read()
                log("文本文件读取成功")
            except Exception as e:
                log(f"读取文件失败: {e}")
                yield "persistent_error", 0, f"**读取文件失败**\n\n`{e}`"
                save_run_log("failed")
                return

            yield "transcript", full_transcript

            if transcript_only:
                transcript_output_path = os.path.join(final_output_dir, f"{output_filename}.txt")
                try:
                    with open(transcript_output_path, 'w', encoding='utf-8') as f:
                        f.write(full_transcript)
                    log(f"转录稿已保存: {transcript_output_path}")
                except IOError as e:
                    log(f"转录稿保存失败: {e}")
                    yield "persistent_error", 0, f"**保存转录稿失败**\n\n`{e}`"
                    save_run_log("failed")
                    return
                current_progress += 1
                log("流程完成（仅转录）")
                task_completed = True
                save_run_log("success")
                yield "progress", current_progress / total_steps, "处理完成！"
                yield "done", transcript_output_path, "✅ 语音转文字稿已生成！"
                return

            current_progress += 1
            yield "progress", current_progress / total_steps, f"步骤 2/2: 正在提交给 {LLM_CONFIG.get('provider_name', 'LLM')}..."
            
            final_path = None
            for event_type, value, *rest in run_llm_and_yield_results():
                if event_type == "persistent_error":
                    yield event_type, value, rest[0]
                    return
                elif event_type == "llm_chunk":
                    yield event_type, value
                elif event_type == "progress_text":
                    yield "progress", current_progress / total_steps, value
                elif event_type == "save_path":
                    final_path = value
                    
            if final_path:
                current_progress += 1
                log("流程完成")
                task_completed = True
                save_run_log("success")
                yield "progress", current_progress / total_steps, "处理完成！"
                yield "done", final_path, "🎉 恭喜！智能笔记已生成！"
            return

        elif file_ext in VIDEO_EXTS or file_ext in AUDIO_EXTS:
            is_video = file_ext in VIDEO_EXTS
            do_visual_analysis = is_video and enable_visual_analysis and not transcript_only
            if transcript_only:
                total_steps = 2
            elif do_visual_analysis:
                total_steps = 4
            else:
                total_steps = 3
            
            step_name = "视频" if is_video else "音频"
            yield "progress", current_progress / total_steps, f"步骤 {current_progress + 1}/{total_steps}: 正在切分{step_name}为音频块..."
            log("开始媒体切分...")

            if transcription_provider == "Qwen API":
                chunk_duration_seconds = 170 
                yield "sub_progress", 0.0, f"使用 Qwen API"
            else:
                chunk_duration_seconds = 720
                yield "sub_progress", 0.0, f"使用 Local Whisper"

            audio_chunks = []
            for event_type, val1, *val2 in split_media_to_audio_chunks_generator(input_path, output_dir, chunk_duration_seconds):
                if event_type == 'progress':
                    yield "sub_progress", val1 / val2[0], f"正在切分... ({val1}/{val2[0]})"
                elif event_type == 'result':
                    audio_chunks = val1
                elif event_type == 'error':
                    log(f"切分失败: {val1}")
                    yield "persistent_error", 0, f"**媒体文件切分失败**\n\n`{val1}`"
                    save_run_log("failed")
                    return
            
            if not audio_chunks:
                log("切分未生成音频块")
                yield "persistent_error", 0, f"**{step_name}切分失败**\n\n未能提取出音频块。"
                save_run_log("failed")
                return
            
            log(f"切分完成，共 {len(audio_chunks)} 块")
            yield "sub_progress", 1.0, f"✅ {step_name}切分全部完成！"
            current_progress += 1
            yield "progress", current_progress / total_steps, f"✅ {step_name}切分完成，准备开始转录..."

            if transcription_provider == "Local Whisper":
                try:
                    yield "sub_progress", 0.0, f"正在准备 Whisper 转录模型 ({whisper_model_size})..."
                    pre_download_whisper_model(whisper_model_size)
                    yield "sub_progress", 1.0, f"✅ Whisper 模型 ({whisper_model_size}) 准备就绪。"
                except Exception as e:
                    log(f"Whisper 模型加载失败: {e}")
                    yield "persistent_error", 0, f"**Whisper 模型加载失败**\n\n`{e}`"
                    save_run_log("failed")
                    return
            else:
                yield "sub_progress", 1.0, f"✅ Qwen API 准备就绪。"

            yield "progress", current_progress / total_steps, f"步骤 {current_progress + 1}/{total_steps}: 正在并行转录 {len(audio_chunks)} 个音频块..."
            log("开始并发转录...")
            all_transcripts = [None] * len(audio_chunks)
            num_transcribed = 0

            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=asr_concurrency) as executor:
                    if transcription_provider == "Local Whisper":
                        future_to_index = {
                            executor.submit(transcribe_single_audio_chunk, chunk, whisper_model_size): i
                            for i, chunk in enumerate(audio_chunks)
                        }
                    else:
                        future_to_index = {
                            executor.submit(transcribe_with_qwen, chunk, asr_context, qwen_asr_model): i
                            for i, chunk in enumerate(audio_chunks)
                        }
                    
                    for future in concurrent.futures.as_completed(future_to_index):
                        index = future_to_index[future]
                        result = future.result() 
                        if result is not None:
                            all_transcripts[index] = result
                        else:
                            raise Exception(f"转录任务失败 (块 {index})")
                        
                        num_transcribed += 1
                        yield "sub_progress", num_transcribed / len(audio_chunks), f"正在转录... ({num_transcribed}/{len(audio_chunks)})"

            except Exception as e:
                log(f"转录异常: {e}")
                yield "persistent_error", 0, f"**音频转录失败 ({transcription_provider})**\n\n`{e}`"
                save_run_log("failed")
                return
            
            if any(t is None for t in all_transcripts):
                log("转录结果不完整")
                yield "persistent_error", 0, "**音频转录不完整**"
                save_run_log("failed")
                return

            log("转录完成")
            yield "sub_progress", 1.0, "✅ 音频转录全部完成！"
            current_progress += 1
            yield "progress", current_progress / total_steps, "所有音频块转录完成！"
            shutil.rmtree(output_dir, ignore_errors=True)

            raw_audio_transcript = "\n\n".join(filter(None, all_transcripts))
            full_transcript = raw_audio_transcript

            if do_visual_analysis:
                yield "progress", current_progress / total_steps, f"步骤 {current_progress + 1}/{total_steps}: 正在进行视频视觉内容分析 (VLM)..."
                log("开始视觉分析...")
                
                try:
                    yield "sub_progress", 0.0, f"正在预处理与抽帧..."
                    
                    image_assets_dir = os.path.join(final_output_dir, "assets")

                    vm = VisualManager()
                    
                    visual_report = ""
                    vm_gen = vm.process_video_generator(
                        input_path, 
                        interval_seconds=60, 
                        mode="lecture_mixed",
                        max_workers=vlm_concurrency,
                        model=vlm_model_name,
                        backup_model=vlm_backup_model,
                        keep_intermediate_files=keep_visual_files,
                        insert_images=insert_images,
                        image_output_dir=image_assets_dir
                    )
                    
                    for vm_event, vm_val1, *vm_args in vm_gen:
                        if vm_event == "stage_progress":
                            stage, curr, total = vm_val1, vm_args[0], vm_args[1]
                            progress_val = curr / total if total > 0 else 0
                            if stage == "extracting":
                                yield "sub_progress", progress_val, f"正在并发抽帧 ({curr}/{total})..."
                            elif stage == "analyzing":
                                yield "sub_progress", progress_val, f"正在分析关键帧 ({curr}/{total})..."
                        elif vm_event == "log":
                             log(f"[VisualManager] {vm_val1}")
                             yield "sub_progress", 0.0, vm_val1
                        elif vm_event == "result":
                             visual_report = vm_val1
                    
                    yield "sub_progress", 1.0, "✅ 视觉分析完成。"
                    
                    instruction_text = "请将以下视觉信息与音频内容结合，生成完整的笔记。"
                    if insert_images:
                        instruction_text += "\n**重要提示**：视觉报告中包含了形如 `![关键帧截图](assets/...)` 的图片链接。请务必根据内容上下文，将这些图片链接**原样插入**到生成的 Markdown 笔记中对应的段落之后，以图文并茂地展示内容。"

                    full_transcript = (
                        "【以下是视频的音频转录内容】\n"
                        f"{raw_audio_transcript}\n\n"
                        "========================================\n"
                        "【以下是视频画面的视觉分析报告（包含PPT/板书内容）】\n"
                        f"{instruction_text}\n\n"
                        f"{visual_report}"
                    )
                    
                    if keep_visual_files:
                        v_report_path = os.path.join(final_output_dir, f"{output_filename}_visual_report.md")
                        try:
                            with open(v_report_path, "w", encoding="utf-8") as f:
                                f.write(visual_report)
                            log(f"视觉报告已保存: {v_report_path}")
                        except Exception as e:
                            log(f"保存视觉报告失败: {e}")

                    current_progress += 1
                    yield "progress", current_progress / total_steps, "视觉分析完成，正在汇总所有内容..."

                except Exception as e:
                    log(f"视觉分析失败: {e}")
                    yield "sub_progress", 1.0, f"⚠️ 视觉分析遇到问题，将仅使用音频内容继续。"
                    full_transcript = raw_audio_transcript
            
            try:
                with open(transcript_save_path, 'w', encoding='utf-8') as f:
                    f.write(full_transcript)
            except IOError:
                pass

            yield "transcript", full_transcript

            if transcript_only:
                transcript_output_path = os.path.join(final_output_dir, f"{output_filename}.txt")
                try:
                    with open(transcript_output_path, 'w', encoding='utf-8') as f:
                        f.write(full_transcript)
                    log(f"转录稿已保存: {transcript_output_path}")
                except IOError as e:
                    log(f"转录稿保存失败: {e}")
                    yield "persistent_error", 0, f"**保存转录稿失败**\n\n`{e}`"
                    save_run_log("failed")
                    return
                current_progress += 1
                log("流程完成（仅转录）")
                task_completed = True
                save_run_log("success")
                yield "progress", current_progress / total_steps, "处理完成！"
                yield "done", transcript_output_path, "✅ 语音转文字稿已生成！"
                return

            yield "progress", current_progress / total_steps, f"步骤 {current_progress + 1}/{total_steps}: 正在提交给 {LLM_CONFIG.get('provider_name', 'LLM')}..."

            final_path = None
            for event_type, value, *rest in run_llm_and_yield_results():
                if event_type == "persistent_error":
                    yield event_type, value, rest[0]
                    return
                elif event_type == "llm_chunk":
                    yield event_type, value
                elif event_type == "progress_text":
                    yield "progress", current_progress / total_steps, value
                elif event_type == "save_path":
                    final_path = value

            if final_path:
                current_progress += 1
                log("流程完成")
                task_completed = True
                save_run_log("success")
                yield "progress", current_progress / total_steps, "处理完成！"
                yield "done", final_path, "🎉 恭喜！智能笔记已生成！"
            return
            
        else:
            log(f"不支持的文件类型: {file_ext}")
            yield "persistent_error", 0, f"**不支持的文件类型**\n\n`{file_ext}`"
            save_run_log("failed")
            return
    except GeneratorExit:
        log("GeneratorExit: 用户主动停止或Streamlit关闭中断")
        if not task_completed:
            save_run_log("stopped_by_user")
        raise
    except Exception as e:
        log(f"未捕获的全局异常: {e}")
        save_run_log("crashed")
        raise