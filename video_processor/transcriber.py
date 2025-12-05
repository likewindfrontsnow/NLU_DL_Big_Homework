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
            print(f"  > (线程 {threading.current_thread().name}) 正在获取加载锁...")
            model = whisper.load_model(model_name)
            print(f"  > (线程 {threading.current_thread().name}) 已释放加载锁。")
        
        print(f"  > ✅ 本地模型 '{model_name}' (线程 {threading.current_thread().name}) 加载成功！")
        return model
    except Exception as e:
        print(f"  > 错误：无法加载本地 Whisper 模型 '{model_name}'。请确保已正确安装 whisper 库及其依赖。")
        print(f"  > 详细错误: {e}")
        raise

def transcribe_single_audio_chunk(audio_path: str, model_name: str) -> str | None:
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
    print(f"  > [Whisper] 正在转录: {audio_filename} (模型: {model_name}, 线程: {threading.current_thread().name})")
    
    try:
        if not os.path.exists(audio_path):
             raise FileNotFoundError
        
        result = model.transcribe(audio_path)
        transcription = result["text"]
        print(f"  > ✅ [Whisper] 文件 '{audio_filename}' 转录成功！")
        return transcription
    
    except FileNotFoundError:
        print(f"  > 错误：找不到音频文件: {audio_path}")
        return None 
    except Exception as e:
        print(f"  > [Whisper] 本地转录失败 (文件: {audio_filename}, 模型: {model_name}): {e}")
        return None

QWEN_RETRY_EXCEPTIONS = (
    subprocess.CalledProcessError, 
    ConnectionError,
    Exception 
)

def _transcribe_sync_qwen(api_key, model_name, audio_path, context_text):
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
        asr_options={
            "enable_itn": True 
        }
    )
    
    if response.status_code == 200 and response.output:
        content_list = response.output.choices[0].message.content
        for item in content_list:
            if "text" in item:
                return item["text"]
        return ""
    else:
        raise Exception(f"Qwen ASR Sync API Error: {response.status_code} - {response.message}")

def _transcribe_async_filetrans(api_key, model_name, audio_path, context_text):
    abs_audio_path = os.path.abspath(audio_path)
    audio_url = f"file://{abs_audio_path}"
    
    try:
        task_response = Transcription.async_call(
            api_key=api_key,
            model=model_name,
            file_urls=[audio_url],
            enable_itn=True
        )
        
        transcription_response = Transcription.wait(task=task_response, api_key=api_key)
        
        if transcription_response.status_code == 200:
            results = transcription_response.output.get("results", [])
            if results:
                result = results[0]
                if result.get("subtask_status") == "SUCCEEDED":
                    trans_url = result.get("transcription_url")
                    if trans_url:
                        r = requests.get(trans_url)
                        data = r.json()
                        if "transcripts" in data:
                            full_text = "".join([t.get("text", "") for t in data["transcripts"]])
                            return full_text
                        elif "text" in data:
                             return data["text"]
                        
                    if "text" in result:
                        return result["text"]
                else:
                    raise Exception(f"Subtask failed: {result.get('message')}")
            return ""
        else:
             raise Exception(f"Transcription Task Failed: {transcription_response.code} - {transcription_response.message}")

    except Exception as e:
        raise Exception(f"Async Transcription Error: {e}")

@retry(max_retries=3, delay=5, allowed_exceptions=QWEN_RETRY_EXCEPTIONS)
def _execute_transcription(api_key, model_name, audio_path, context):
    """
    根据模型名称自动判断使用同步还是异步接口
    """
    # 简单的判断逻辑：文件名包含 'filetrans' 或 'fun-asr' 则为异步
    is_async = "filetrans" in model_name or "fun-asr" in model_name
    
    if is_async:
        return _transcribe_async_filetrans(api_key, model_name, audio_path, context)
    else:
        return _transcribe_sync_qwen(api_key, model_name, audio_path, context)

@retry(max_retries=5, delay=2, backoff_factor=2, allowed_exceptions=QWEN_RETRY_EXCEPTIONS)
def transcribe_with_qwen(audio_path: str, asr_context: str | None = None, model_name: str = "qwen3-asr-flash") -> str | None:
    current_api_key = os.getenv("DASHSCOPE_API_KEY") or LLM_CONFIG.get("api_key")
    backup_model = LLM_CONFIG.get("asr_backup_model") # 从配置获取备选模型
    
    context_text = asr_context if asr_context else ""

    if not current_api_key:
        raise Exception("DASHSCOPE_API_KEY (LLM_API_KEY) 未设置")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    # 构建尝试列表：[主模型, 备选模型]
    models_to_try = [model_name]
    if backup_model and backup_model != model_name:
        models_to_try.append(backup_model)
    
    # 遍历尝试
    for i, current_model in enumerate(models_to_try):
        try:
            # 调用辅助函数执行转录
            result = _execute_transcription(current_api_key, current_model, audio_path, context_text)
            
            if result is not None:
                return result
            
        except Exception as e:
            # 检查是否为限流相关错误
            err_str = str(e).lower()
            is_rate_limit = "429" in err_str or "quota" in err_str or "rate" in err_str
            
            # 如果是限流错误，且还有下一个模型可试，则切换
            if is_rate_limit:
                if i < len(models_to_try) - 1:
                    # 可以在这里加一个简单的 print 提示切换，或者直接 continue
                    time.sleep(1) # 稍作缓冲
                    continue
            
            # 如果不是限流错误，或者已经没有备用模型，则抛出异常
            # 这会触发 @retry 装饰器进行指数退避重试
            raise e

    return None