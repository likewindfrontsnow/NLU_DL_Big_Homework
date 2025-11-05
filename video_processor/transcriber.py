# transcriber.py
import os
import whisper
import threading
import dashscope
from config import LLM_API_KEY
DASHSCOPE_API_KEY=LLM_API_KEY
from utils import retry
import subprocess 

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
            print(f"  > (线程 {threading.current_thread().name})  đang获取加载锁...")
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

@retry(max_retries=3, delay=5, allowed_exceptions=QWEN_RETRY_EXCEPTIONS)
def transcribe_with_qwen(audio_path: str, asr_context: str | None = None) -> str | None:
    audio_filename = os.path.basename(audio_path)
    print(f"  > [Qwen API] 正在提交: {audio_filename} (线程: {threading.current_thread().name})")
    if asr_context:
        print(f"  > [Qwen API] 使用上下文增强: {asr_context[:50]}...")

    try:
        if not DASHSCOPE_API_KEY:
            print("  > 错误: DASHSCOPE_API_KEY (LLM_API_KEY) 未在 .env 中设置。")
            raise Exception("DASHSCOPE_API_KEY (LLM_API_KEY) 未在 .env 中设置。")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        abs_audio_path = os.path.abspath(audio_path)
        
        context_text = asr_context if asr_context else ""
        
        messages = [
            {
                "role": "system",
                "content": [
                    {"text": context_text},
                ]
            },
            {
                "role": "user",
                "content": [
                    {"audio": f"file://{abs_audio_path}"}, 
                ]
            }
        ]
        dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

        response = dashscope.MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model="qwen3-asr-flash", 
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
                    transcription = item["text"]
                    print(f"  > ✅ [Qwen API] 文件 '{audio_filename}' 转录成功！")
                    return transcription
            
            print(f"  > 错误 [Qwen API]: API 响应成功，但在 content 中未找到 'text'。 响应: {response}")
            return None
        else:
            print(f"  > 错误 [Qwen API]: API 调用失败 (文件: {audio_filename})。")
            print(f"  > 状态码: {response.status_code}, 响应: {response}")
            raise Exception(f"Qwen ASR API 错误: {response.status_code} - {response.message}")

    except FileNotFoundError:
        print(f"  > 错误 [Qwen API]: 找不到音频文件: {audio_path}")
        return None
    except Exception as e:
        print(f"  > [Qwen API] 转录失败 (文件: {audio_filename}): {e}")
        raise e