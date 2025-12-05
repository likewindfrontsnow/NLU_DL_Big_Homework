# video_processor/transcriber.py
import os
import whisper
import threading
import dashscope
from dashscope.audio.asr import Transcription
from config import LLM_CONFIG
from utils import retry
import subprocess
import json
import requests
import time

MODEL_STORAGE = threading.local()
_DOWNLOAD_LOCK = threading.Lock()
_LOAD_LOCK = threading.Lock()

def _seconds_to_timestamp(seconds: float) -> str:
    """将秒数转换为 MM:SS 格式"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

def pre_download_whisper_model(model_name: str):
    with _DOWNLOAD_LOCK:
        print(f"  > 正在检查并预下载 Whisper 模型: {model_name}...")
        try:
            _ = whisper.load_model(model_name)
            print(f"  > ✅ 模型 '{model_name}' 已准备就绪 (已缓存或已下载)。")
            return True
        except Exception as e:
            print(f"  > 严重错误：无法下载/加载 Whisper 模型 '{model_name}'。")
            print(f"  > 详细错误: {e}")
            raise

def _load_whisper_model(model_name: str):
    print(f"  > 正在为当前线程加载本地 Whisper 模型: {model_name}...")
    try:
        with _LOAD_LOCK:
            model = whisper.load_model(model_name)
        print(f"  > ✅ 本地模型 '{model_name}' (线程 {threading.current_thread().name}) 加载成功！")
        return model
    except Exception as e:
        print(f"  > 错误：无法加载本地 Whisper 模型 '{model_name}'。")
        raise

def transcribe_single_audio_chunk(audio_path: str, model_name: str, start_offset_seconds: int = 0) -> str | None:
    """
    Local Whisper 转录函数
    参数 start_offset_seconds 用于计算全局绝对时间
    """
    model_loaded_correctly = (
        hasattr(MODEL_STORAGE, "model") and 
        hasattr(MODEL_STORAGE, "model_name") and
        MODEL_STORAGE.model_name == model_name
    )

    if not model_loaded_correctly:
        try:
            MODEL_STORAGE.model = _load_whisper_model(model_name)
            MODEL_STORAGE.model_name = model_name
        except Exception:
            return None 

    model = MODEL_STORAGE.model
    audio_filename = os.path.basename(audio_path)
    print(f"  > [Whisper] 正在转录: {audio_filename} (偏移: {start_offset_seconds}s)")
    
    try:
        if not os.path.exists(audio_path):
             raise FileNotFoundError
        
        # Whisper 原生支持 segments 时间戳
        result = model.transcribe(audio_path)
        formatted_transcript = ""
        
        if "segments" in result:
            for segment in result["segments"]:
                # 绝对时间 = 块起始时间 + 句内相对时间
                absolute_start = segment['start'] + start_offset_seconds
                time_str = _seconds_to_timestamp(absolute_start)
                text = segment['text'].strip()
                if text:
                    formatted_transcript += f"[{time_str}] {text}\n"
        else:
            # 兜底
            time_str = _seconds_to_timestamp(start_offset_seconds)
            formatted_transcript = f"[{time_str}] {result['text']}\n"
            
        print(f"  > ✅ [Whisper] 文件 '{audio_filename}' 转录成功！")
        return formatted_transcript
    
    except Exception as e:
        print(f"  > [Whisper] 转录失败: {e}")
        return None

QWEN_RETRY_EXCEPTIONS = (
    subprocess.CalledProcessError, 
    ConnectionError,
    Exception 
)

def _transcribe_sync_qwen(api_key, model_name, audio_path, context_text, start_offset_seconds):
    """
    Qwen 同步接口 (qwen3-asr-flash)
    注意：同步模型不支持返回时间戳，仅在段首标记近似开始时间。
    """
    abs_audio_path = os.path.abspath(audio_path)
    
    messages = []
    if context_text and context_text.strip():
        messages.append({
            "role": "system",
            "content": [{"text": context_text}]
        })
    
    messages.append({
        "role": "user",
        "content": [{"audio": f"file://{abs_audio_path}"}]
    })

    response = dashscope.MultiModalConversation.call(
        api_key=api_key,
        model=model_name, 
        messages=messages,
        result_format="message",
        asr_options={"enable_itn": True}
    )
    
    if response.status_code == 200 and response.output:
        content_list = response.output.choices[0].message.content
        text_result = ""
        for item in content_list:
            if "text" in item:
                text_result += item["text"]
        
        if text_result:
            # 仅在段首加一个时间戳作为近似定位
            time_str = _seconds_to_timestamp(start_offset_seconds)
            return f"[{time_str}] {text_result}\n"
        return ""
    else:
        raise Exception(f"Qwen API Error: {response.code} - {response.message}")

def _transcribe_async_filetrans(api_key, model_name, audio_path, context_text, start_offset_seconds):
    """
    Qwen 异步接口 (qwen3-asr-flash-filetrans)
    支持解析 sentences 列表中的 begin_time (毫秒)
    """
    abs_audio_path = os.path.abspath(audio_path)
    audio_url = f"file://{abs_audio_path}"
    
    try:
        # 1. 提交任务
        task_response = Transcription.async_call(
            api_key=api_key,
            model=model_name,
            file_urls=[audio_url],
            enable_itn=True
            # 注意：如果 SDK 支持在此处传 text 上下文，可添加 parameters={'text': context_text}
            # 具体取决于 SDK 版本，此处按标准调用
        )
        
        # 2. 等待任务完成
        transcription_response = Transcription.wait(task=task_response, api_key=api_key)
        
        if transcription_response.status_code == 200:
            results = transcription_response.output.get("results", [])
            if results:
                result = results[0]
                if result.get("subtask_status") == "SUCCEEDED":
                    trans_url = result.get("transcription_url")
                    if trans_url:
                        # 3. 下载并解析详细结果 JSON
                        r = requests.get(trans_url)
                        data = r.json()
                        
                        # --- 解析逻辑 ---
                        if "transcripts" in data:
                            full_text_with_ts = ""
                            for t in data["transcripts"]:
                                # 优先尝试解析 sentences
                                sentences = t.get("sentences", [])
                                if sentences:
                                    for sent in sentences:
                                        # begin_time 单位是毫秒 [cite: 18]
                                        begin_ms = sent.get("begin_time", 0)
                                        # 转换为绝对秒数：(毫秒/1000) + 音频块起始偏移
                                        abs_seconds = (begin_ms / 1000.0) + start_offset_seconds
                                        ts_str = _seconds_to_timestamp(abs_seconds)
                                        
                                        text_content = sent.get('text', '')
                                        full_text_with_ts += f"[{ts_str}] {text_content}\n"
                                else:
                                    # 如果没有 sentences 字段，回退到整段文本
                                    ts_str = _seconds_to_timestamp(start_offset_seconds)
                                    full_text_with_ts += f"[{ts_str}] {t.get('text', '')}\n"
                            
                            return full_text_with_ts
                        
                        # 兜底逻辑：只有 text 字段时
                        if "text" in data:
                             ts_str = _seconds_to_timestamp(start_offset_seconds)
                             return f"[{ts_str}] {data['text']}\n"
                    
                    # 极少数情况 result 中直接有 text
                    if "text" in result:
                        ts_str = _seconds_to_timestamp(start_offset_seconds)
                        return f"[{ts_str}] {result['text']}\n"
                else:
                    raise Exception(f"Subtask failed: {result.get('message')}")
            return ""
        else:
             raise Exception(f"Task Failed: {transcription_response.message}")
    except Exception as e:
        raise Exception(f"Async Transcription Error: {e}")

@retry(max_retries=3, delay=5, allowed_exceptions=QWEN_RETRY_EXCEPTIONS)
def _execute_transcription(api_key, model_name, audio_path, context, offset):
    """
    根据模型名称自动判断使用同步还是异步接口，并传入偏移量
    """
    is_async = "filetrans" in model_name or "fun-asr" in model_name
    
    if is_async:
        return _transcribe_async_filetrans(api_key, model_name, audio_path, context, offset)
    else:
        return _transcribe_sync_qwen(api_key, model_name, audio_path, context, offset)

@retry(max_retries=5, delay=2, backoff_factor=2, allowed_exceptions=QWEN_RETRY_EXCEPTIONS)
def transcribe_with_qwen(audio_path: str, asr_context: str | None = None, model_name: str = "qwen3-asr-flash", start_offset_seconds: int = 0) -> str | None:
    """
    Qwen 转录入口，支持 start_offset_seconds
    """
    current_api_key = os.getenv("DASHSCOPE_API_KEY") or LLM_CONFIG.get("api_key")
    backup_model = LLM_CONFIG.get("asr_backup_model")
    
    context_text = asr_context if asr_context else ""

    if not current_api_key:
        raise Exception("DASHSCOPE_API_KEY 未设置")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    models_to_try = [model_name]
    if backup_model and backup_model != model_name:
        models_to_try.append(backup_model)
    
    for i, current_model in enumerate(models_to_try):
        try:
            result = _execute_transcription(current_api_key, current_model, audio_path, context_text, start_offset_seconds)
            
            if result is not None:
                return result
            
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "quota" in err_str or "rate" in err_str
            
            if is_rate_limit:
                if i < len(models_to_try) - 1:
                    time.sleep(1)
                    continue
            raise e

    return None