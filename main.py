# main.py
import concurrent.futures
import time
import os
import sys
import shutil
# (修改) 导入新的通用配置
from config import LLM_CONFIG
from video_processor.splitter import split_media_to_audio_chunks_generator
from video_processor.transcriber import transcribe_single_audio_chunk
# (修改) 导入新的通用 LLM API 函数
from llm_api import run_llm_generation 

def main_process_generator(input_path: str, dify_api_key: str, output_filename: str, query: str):
    """
    - 一个生成器函数，执行处理流程并实时产出状态、进度和LLM文本块。
    - (已修改) 移除 Dify 依赖，转而调用通用的 llm_api.py
    """
    # (注意: dify_api_key 参数现在已未使用，但为了保持 app.py 调用不变，暂且保留)
    
    output_dir = "output_chunks"
    final_notes_save_path = f"{output_filename}.md"
    
    video_exts = {'.mp4', '.mov', '.mpeg', '.webm'}
    audio_exts = {'.mp3', '.m4a', '.wav', '.amr', '.mpga'}
    text_exts = {'.txt', '.md', '.mdx', '.markdown', '.pdf', '.html', '.xlsx', '.xls', '.doc', '.docx', '.csv', '.eml', '.msg', '.pptx', '.ppt', '.xml', '.epub'}

    file_ext = os.path.splitext(input_path)[1].lower()
    current_progress = 0
    full_transcript = ""

    # --- (重构) 移除了 Dify 依赖的辅助函数 ---
    def run_llm_and_yield_results():
        """辅助生成器：运行 LLM 并处理结果。"""
        
        try:
            # (修改) 动态显示 LLM 服务商和模型
            provider_name = LLM_CONFIG.get('provider_name', 'LLM')
            model_name = LLM_CONFIG.get('model', 'default')
            yield "progress_text", f"正在提交给 {provider_name} (模型: {model_name})..."
            
            # --- 核心调用 (修改) ---
            # 这现在是一个阻塞调用，会等待 LLM 完整响应
            final_text = run_llm_generation(full_transcript, query) # <-- (修改)
            
            if not final_text:
                # (修改) 通用错误信息
                yield "persistent_error", 0, "**笔记生成失败**\n\nAPI 在多次尝试后，未返回任何有效内容。请检查您的 API 配置以及输入文本是否过长或格式异常。"
                return

            # --- 模拟流式输出 ---
            yield "llm_chunk", final_text
            
            # --- 保存文件 (逻辑保留) ---
            try:
                with open(final_notes_save_path, 'w', encoding='utf-8') as f:
                    f.write(final_text)
                yield "save_path", final_notes_save_path
            except IOError as e:
                user_friendly_error = f"**保存最终笔记文件失败**\n\n无法将生成的笔记写入本地文件。\n\n**可能原因:**\n- 程序没有在当前目录创建文件的权限。\n- 磁盘空间不足。\n\n**原始错误信息:**\n`{e}`"
                yield "persistent_error", 0, user_friendly_error
                return

        except ValueError as e: # 捕获 (query != "Notes") 的错误
            yield "persistent_error", 0, str(e)
            return
        except Exception as e: # 捕获 API 调用失败 (重试后)
            # (修改) 动态显示错误源
            provider_name = LLM_CONFIG.get('provider_name', 'LLM')
            user_friendly_error = f"**笔记生成失败**\n\n看起来在与 {provider_name} API 服务通信时遇到了问题。\n\n**原始错误信息:**\n`{e}`"
            yield "persistent_error", 0, user_friendly_error
            return


    # === 文本文件工作流 ===
    if file_ext in text_exts:
        total_steps = 2
        yield "progress", 0 / total_steps, "步骤 1/2: 正在读取文本文档..."
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                full_transcript = f.read()
        except Exception as e:
            user_friendly_error = f"**读取文件失败**\n\n无法读取您上传的文本文档 '{os.path.basename(input_path)}'。\n\n**可能原因:**\n- 文件已损坏或编码格式不是 UTF-8。\n- 程序没有读取该文件的权限。\n\n**原始错误信息:**\n`{e}`"
            yield "persistent_error", 0, user_friendly_error
            return
        
        current_progress += 1
        # (修改) 动态显示提交目标
        provider_name = LLM_CONFIG.get('provider_name', 'LLM')
        yield "progress", current_progress / total_steps, f"步骤 2/2: 正在提交给 {provider_name}..."
        
        final_path = None
        # --- (修改) 使用新的辅助函数 ---
        llm_gen = run_llm_and_yield_results()
        for event_type, value, *rest in llm_gen:
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
            yield "progress", current_progress / total_steps, "处理完成！"
            yield "done", final_path, "🎉 恭喜！智能笔记已生成！"
        return

    # === 视频和音频文件工作流 ===
    elif file_ext in video_exts or file_ext in audio_exts:
        # ... (媒体文件切分和转录部分保持不变) ...
        is_video = file_ext in video_exts
        total_steps = 4 if is_video else 3
        
        step_name = "视频" if is_video else "音频"
        yield "progress", current_progress / total_steps, f"步骤 {current_progress + 1}/{total_steps}: 正在切分{step_name}为音频块..."
        
        splitter_generator = split_media_to_audio_chunks_generator(input_path, output_dir, 600)
        audio_chunks = []
        
        for event_type, val1, *val2 in splitter_generator:
            if event_type == 'progress':
                completed, total = val1, val2[0]
                yield "sub_progress", completed / total, f"正在切分... ({completed}/{total})"
            elif event_type == 'result':
                audio_chunks = val1
            elif event_type == 'error':
                user_friendly_error = f"**媒体文件切分失败**\n\n无法处理您上传的媒体文件。这通常与 **FFmpeg** 配置或文件本身有关。\n\n**请检查:**\n1. **FFmpeg 是否已正确安装**: 确保 FFmpeg 已安装并在系统的环境变量 `PATH` 中。\n2. **文件是否完好**: 确认您的文件 `{os.path.basename(input_path)}` 没有损坏且格式受支持。\n\n**原始错误信息:**\n`{val1}`"
                yield "persistent_error", 0, user_friendly_error
                return
        
        if not audio_chunks:
            yield "persistent_error", 0, f"**{step_name}切分失败**\n\n未能从您的文件中提取出任何音频块。请确保文件时长不为零，且已正确安装 FFmpeg。"
            return
        
        yield "sub_progress", 1.0, f"✅ {step_name}切分全部完成！"
        current_progress += 1
        yield "progress", current_progress / total_steps, f"✅ {step_name}切分完成，准备开始转录..."

        yield "progress", current_progress / total_steps, f"步骤 {current_progress + 1}/{total_steps}: 正在并行转录 {len(audio_chunks)} 个音频块..."
        all_transcripts = [None] * len(audio_chunks)
        num_transcribed = 0

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_index = {
                    executor.submit(transcribe_single_audio_chunk, chunk): i
                    for i, chunk in enumerate(audio_chunks)
                }
                
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    result = future.result() 
                    if result is not None:
                        all_transcripts[index] = result
                    else:
                        raise Exception(f"本地转录任务未返回有效文本 (块索引: {index})。请检查 Whisper 模型是否正确加载。")
                    
                    num_transcribed += 1
                    yield "sub_progress", num_transcribed / len(audio_chunks), f"正在转录... ({num_transcribed}/{len(audio_chunks)})"

        except Exception as e:
            user_friendly_error = f"**音频转录失败**\n\n在本地使用 Whisper 进行语音转文字时发生无法恢复的错误。\n\n**可能原因:**\n1. **Whisper 模型加载失败**: 确保 `tiny` 模型文件可访问。\n2. **依赖库问题**: 确保 `openai-whisper` 及其依赖（如 PyTorch）已正确安装。\n3. **音频数据问题**: 某个音频块可能已损坏无法处理。\n\n**原始错误信息:**\n`{e}`"
            yield "persistent_error", 0, user_friendly_error
            return
        
        if any(t is None for t in all_transcripts):
            yield "persistent_error", 0, "**音频转录不完整**\n\n部分音频块在多次尝试后仍然转录失败。为确保笔记的完整性，处理已中止。"
            return

        yield "sub_progress", 1.0, "✅ 音频转录全部完成！"
        current_progress += 1
        yield "progress", current_progress / total_steps, "所有音频块转录完成！"
        shutil.rmtree(output_dir, ignore_errors=True)

        if is_video:
            yield "progress", current_progress / total_steps, f"步骤 {current_progress + 1}/{total_steps}: 正在汇总文字稿并保存..."
        
        full_transcript = "\n\n".join(filter(None, all_transcripts))
        
        transcript_save_path = "source_transcript.txt"
        try:
            with open(transcript_save_path, 'w', encoding='utf-8') as f:
                f.write(full_transcript)
        except IOError as e:
            yield "error", 0, f"无法保存文字稿文件: {e}"

        if is_video:
            current_progress += 1
            yield "progress", current_progress / total_steps, "文字稿汇总完成。"
            
        # (修改) 动态显示提交目标
        provider_name = LLM_CONFIG.get('provider_name', 'LLM')
        yield "progress", current_progress / total_steps, f"步骤 {current_progress + 1}/{total_steps}: 正在提交给 {provider_name}..."

        final_path = None
        # --- (修改) 使用新的辅助函数 ---
        llm_gen = run_llm_and_yield_results()
        for event_type, value, *rest in llm_gen:
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
            yield "progress", current_progress / total_steps, "处理完成！"
            yield "done", final_path, "🎉 恭喜！智能笔记已生成！"
        return
        
    else:
        user_friendly_error = f"**不支持的文件类型**\n\n您上传的文件类型 (`{file_ext}`) 当前不受支持。请参照上传框下的提示，上传指定格式的视频、音频或文本文档。"
        yield "error", 0, user_friendly_error
        return