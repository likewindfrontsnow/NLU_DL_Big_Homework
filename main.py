# main.py
import concurrent.futures
import time
import os
import sys
import shutil
from config import LLM_CONFIG
from video_processor.splitter import split_media_to_audio_chunks_generator
from video_processor.transcriber import transcribe_single_audio_chunk, pre_download_whisper_model, transcribe_with_qwen
from llm_api import run_llm_generation, refine_llm_generation 

# 生成器函数，处理流程并实时产出进度与LLM文本块
def main_process_generator(input_path: str,  output_filename: str, whisper_model_size: str, stream_output: bool, transcription_provider: str, note_type: str, asr_context: str | None = None, additional_instructions: str = ""): 
    # 一些初始化工作
    output_dir = "output_chunks"
    final_notes_save_path = f"{output_filename}.md"
    
    video_exts = {'.mp4', '.mov', '.mpeg', '.webm'}
    audio_exts = {'.mp3', '.m4a', '.wav', '.amr', '.mpga'}
    text_exts = {'.txt', '.md', '.mdx', '.markdown', '.pdf', '.html', '.xlsx', '.xls', '.doc', '.docx', '.csv', '.eml', '.msg', '.pptx', '.ppt', '.xml', '.epub'}

    file_ext = os.path.splitext(input_path)[1].lower()
    current_progress = 0
    full_transcript = ""

    # 运行llm并处理结果
    def run_llm_and_yield_results():
        
        final_text = "" 
        
        try:
            provider_name = LLM_CONFIG.get('provider_name', 'LLM')
            model_name = LLM_CONFIG.get('model', 'default')
            stream_status = "流式" if stream_output else "非流式"
            yield "progress_text", f"正在提交给 {provider_name} (模型: {model_name}, 模式: {stream_status}, 类型: {note_type})..."
            
            # 转录稿full_transcript传输给llm生成笔记
            llm_call_result = run_llm_generation(full_transcript, stream_output, note_type, additional_instructions)
            
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
                yield "persistent_error", 0, "**笔记生成失败**\n\nAPI 在多次尝试后，未返回任何有效内容。请检查您的 API 配置以及输入文本是否过长或格式异常。"
                return

            try:
                with open(final_notes_save_path, 'w', encoding='utf-8') as f:
                    f.write(final_text)
                yield "save_path", final_notes_save_path
            except IOError as e:
                user_friendly_error = f"**保存最终笔记文件失败**\n\n无法将生成的笔记写入本地文件。\n\n**可能原因:**\n- 程序没有在当前目录创建文件的权限。\n- 磁盘空间不足。\n\n**原始错误信息:**\n`{e}`"
                yield "persistent_error", 0, user_friendly_error
                return

        except ValueError as e: 
            yield "persistent_error", 0, str(e)
            return
        except Exception as e: 
            provider_name = LLM_CONFIG.get('provider_name', 'LLM')
            user_friendly_error = f"**笔记生成失败**\n\n看起来在与 {provider_name} API 服务通信时遇到了问题。\n\n**原始错误信息:**\n`{e}`"
            yield "persistent_error", 0, user_friendly_error
            return
        
    # === 文本文档工作流 ===    
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
        
        yield "transcript", full_transcript 
        
        current_progress += 1
        provider_name = LLM_CONFIG.get('provider_name', 'LLM')
        yield "progress", current_progress / total_steps, f"步骤 2/2: 正在提交给 {provider_name}..."
        
        final_path = None
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
        is_video = file_ext in video_exts
        total_steps = 4 if is_video else 3
        
        step_name = "视频" if is_video else "音频"
        yield "progress", current_progress / total_steps, f"步骤 {current_progress + 1}/{total_steps}: 正在切分{step_name}为音频块..."
        
        if transcription_provider == "Qwen API":
            chunk_duration_seconds = 170 
            yield "sub_progress", 0.0, f"使用 Qwen API"
        else:
            chunk_duration_seconds = 720
            yield "sub_progress", 0.0, f"使用 Local Whisper"

        splitter_generator = split_media_to_audio_chunks_generator(input_path, output_dir, chunk_duration_seconds) 
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

        if transcription_provider == "Local Whisper":
            try:
                yield "sub_progress", 0.0, f"正在准备 Whisper 转录模型 ({whisper_model_size})..."
                pre_download_whisper_model(whisper_model_size)
                yield "sub_progress", 1.0, f"✅ Whisper 模型 ({whisper_model_size}) 准备就绪。"
            except Exception as e:
                user_friendly_error = f"**Whisper 模型加载失败**\n\n无法下载或加载指定的 Whisper 模型 '{whisper_model_size}'。\n\n**可能原因:**\n1. 网络连接问题（如果模型未缓存）。\n2. 模型名称拼写错误。\n3. 磁盘空间不足或权限问题。\n\n**原始错误信息:**\n`{e}`"
                yield "persistent_error", 0, user_friendly_error
                return
        else:
             yield "sub_progress", 1.0, f"✅ Qwen API 准备就绪。"

        yield "progress", current_progress / total_steps, f"步骤 {current_progress + 1}/{total_steps}: 正在并行转录 {len(audio_chunks)} 个音频块..."
        all_transcripts = [None] * len(audio_chunks)
        num_transcribed = 0

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_index = {}
                if transcription_provider == "Local Whisper":
                    print(f"--- 开始使用 Local Whisper (模型: {whisper_model_size}) 进行转录 ---")
                    future_to_index = {
                        executor.submit(transcribe_single_audio_chunk, chunk, whisper_model_size): i
                        for i, chunk in enumerate(audio_chunks)
                    }
                elif transcription_provider == "Qwen API":
                    print("--- 开始使用 Qwen API (模型: qwen3-asr-flash) 进行转录 ---")
                    future_to_index = {
                        executor.submit(transcribe_with_qwen, chunk, asr_context): i
                        for i, chunk in enumerate(audio_chunks)
                    }

                
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    result = future.result() 
                    if result is not None:
                        all_transcripts[index] = result
                    else:
                        error_msg = f"转录任务未返回有效文本 (块索引: {index}, 服务: {transcription_provider})。"
                        if transcription_provider == "Qwen API":
                            error_msg += " 请检查 API Key 是否正确以及网络连接。"
                        else:
                             error_msg += f" 请检查 Whisper 模型 '{whisper_model_size}' 是否正确加载。"
                        raise Exception(error_msg)
                    
                    num_transcribed += 1
                    yield "sub_progress", num_transcribed / len(audio_chunks), f"正在转录... ({num_transcribed}/{len(audio_chunks)})"

        except Exception as e:
            user_friendly_error = f"**音频转录失败 ({transcription_provider})**\n\n在转录时发生无法恢复的错误。\n\n**可能原因:**\n"
            if transcription_provider == "Qwen API":
                user_friendly_error += "1. **API Key 错误**: 检查 `.env` 中的 `LLM_API_KEY` (Qwen API 正在复用此 Key) 是否正确且有效。\n2. **网络问题**: 无法连接到 DashScope API 服务。\n3. **文件问题**: 某个音频块已损坏无法处理。\n"
            else:
                user_friendly_error += f"1. **Whisper 模型加载失败**: 确保指定的 '{whisper_model_size}' 模型文件可访问。\n2. **依赖库问题**: 确保 `openai-whisper` 及其依赖（如 PyTorch）已正确安装。\n"
            
            user_friendly_error += f"\n**原始错误信息:**\n`{e}`"
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

        yield "transcript", full_transcript 

        if is_video:
            current_progress += 1
            yield "progress", current_progress / total_steps, "文字稿汇总完成。"
            
        provider_name = LLM_CONFIG.get('provider_name', 'LLM')
        yield "progress", current_progress / total_steps, f"步骤 {current_progress + 1}/{total_steps}: 正在提交给 {provider_name}..."

        final_path = None
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


# 用于运行、打印生成器输出
def run_test(test_file_path, provider="", model_size="tiny", context=None, stream=False):
    print(f"\n>>> 正在测试文件: {test_file_path} (服务: {provider})")
    
    if not os.path.exists(test_file_path):
        print(f"!!! 警告: 测试文件 '{test_file_path}' 不存在。跳过此测试。")
        return

    try:
        output_name = os.path.splitext(os.path.basename(test_file_path))[0] + "_test_notes"
        
        generator = main_process_generator(
            input_path=test_file_path,
            output_filename=output_name,
            whisper_model_size=model_size,
            stream_output=stream,
            transcription_provider=provider,
            asr_context=context,
            note_type="STEM"
        )

        final_path = None
        
        for event_type, value, *rest in generator:
            text = rest[0] if rest else ""
            
            if event_type == "progress":
                print(f"[进度] {value*100:.0f}% - {text}")
            elif event_type == "sub_progress":
                print(f"  [子进度] {value*100:.0f}% - {text}")
            elif event_type == "transcript":
                print(f"[转录稿生成] (前100字符): {value[:100]}...")
            elif event_type == "llm_chunk":
                print(f"{value}", end="")
            elif event_type == "persistent_error" or event_type == "error":
                print(f"\n\n!!! [严重错误] {text} !!!")
                break
            elif event_type == "done":
                final_path = value
                print(f"\n\n[完成] {text}")
                print(f"最终文件保存在: {final_path}")
                break
        
        if final_path:
            print(f"\n>>> ✅ 测试 {test_file_path} 成功。")
        else:
            print(f"\n>>> ❌ 测试 {test_file_path} 失败或未完成。")

    except Exception as e:
        print(f"\n!!! [测试时发生意外异常] {e}")
        import traceback
        traceback.print_exc()



# 测试
# if __name__ == "__main__":
#     print("--- [主模块测试] 开始 ---")

# # **测试 1: 文本文档**
# print("--- 开始测试 [文本文档] ---")
# run_test(
#     test_file_path="test_doc.txt",
#     provider="Local Whisper"
# )

# **测试 2: 音频文件 (Local Whisper)**
# print("--- 开始测试 [音频文件 - Local Whisper] ---")
# run_test(
#     test_file_path="test_audio.mp3", 
#     provider="Local Whisper",
#     model_size="tiny",
#     stream=False 
# )

# **测试 3: 音频文件 (Qwen API)**
# print("--- 开始测试 [音频文件 - Qwen API] ---")
# run_test(
#     test_file_path="test_audio.mp3",
#     provider="Qwen API",
#     context="财经, 投行, A股", # 测试上下文
#     stream=False
# )
    
