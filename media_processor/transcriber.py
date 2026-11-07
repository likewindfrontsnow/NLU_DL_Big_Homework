import os
import threading
import time
import requests
import whisper
import dashscope
from core.config import LLM_CONFIG
from core.utils import retry
from http import HTTPStatus

MODEL_STORAGE = threading.local()
_LOAD_LOCK = threading.Lock()

def _seconds_to_timestamp(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

def pre_download_whisper_model(model_name: str):
    with _LOAD_LOCK:
        try:
            whisper.load_model(model_name)
            return True
        except Exception:
            raise

def transcribe_single_audio_chunk(audio_path: str, model_name: str, start_offset_seconds: int = 0) -> str | None:
    if not hasattr(MODEL_STORAGE, "model") or getattr(MODEL_STORAGE, "model_name", None) != model_name:
        try:
            with _LOAD_LOCK:
                MODEL_STORAGE.model = whisper.load_model(model_name)
            MODEL_STORAGE.model_name = model_name
        except Exception:
            return None

    try:
        if not os.path.exists(audio_path):
             return None
        
        result = MODEL_STORAGE.model.transcribe(audio_path)
        formatted_transcript = ""
        
        if "segments" in result:
            for segment in result["segments"]:
                absolute_start = segment['start'] + start_offset_seconds
                formatted_transcript += f"[{_seconds_to_timestamp(absolute_start)}] {segment['text'].strip()}\n"
        else:
            formatted_transcript = f"[{_seconds_to_timestamp(start_offset_seconds)}] {result['text']}\n"
            
        return formatted_transcript
    except Exception:
        return None

def _transcribe_sync_qwen(api_key, model_name, audio_path, context_text, start_offset_seconds):
    abs_audio_path = os.path.abspath(audio_path)
    messages = []
    if context_text and context_text.strip():
        messages.append({"role": "system", "content": [{"text": context_text}]})
    
    messages.append({"role": "user", "content": [{"audio": f"file://{abs_audio_path}"}]})

    response = dashscope.MultiModalConversation.call(
        api_key=api_key,
        model=model_name, 
        messages=messages,
        result_format="message",
        asr_options={"enable_itn": True}
    )
    
    if response.status_code == 200 and response.output:
        text_content = [item["text"] for item in response.output.choices[0].message.content if "text" in item]
        text_result = "".join(text_content)
        if text_result:
            return f"[{_seconds_to_timestamp(start_offset_seconds)}] {text_result}\n"
        return ""
    raise Exception(f"Qwen API Error: {response.code} - {response.message}")

@retry(max_retries=3, delay=5, allowed_exceptions=(Exception,))
def _execute_transcription(api_key, model_name, audio_path, context, offset):
    return _transcribe_sync_qwen(api_key, model_name, audio_path, context, offset)

@retry(max_retries=5, delay=2, backoff_factor=2, allowed_exceptions=(Exception,))
def transcribe_with_qwen(audio_path: str, asr_context: str | None = None, model_name: str = "qwen3-asr-flash", start_offset_seconds: int = 0) -> str | None:
    current_api_key = os.getenv("DASHSCOPE_API_KEY") or LLM_CONFIG.get("api_key")
    if not current_api_key:
        raise Exception("DASHSCOPE_API_KEY 未设置")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    backup = LLM_CONFIG.get("asr_backup_model")
    models_to_try = [model_name]
    if backup and backup != model_name:
        models_to_try.append(backup)
    
    for i, current_model in enumerate(models_to_try):
        try:
            result = _execute_transcription(current_api_key, current_model, audio_path, asr_context or "", start_offset_seconds)
            if result is not None:
                return result
        except Exception as e:
            err_str = str(e).lower()
            if ("429" in err_str or "quota" in err_str) and i < len(models_to_try) - 1:
                time.sleep(1)
                continue
            raise e
    return None