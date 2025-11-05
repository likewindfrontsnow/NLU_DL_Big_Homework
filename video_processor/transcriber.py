# transcriber.py
import os
import whisper
import threading
import dashscope
from config import DASHSCOPE_API_KEY
from utils import retry
import subprocess 

MODEL_STORAGE = threading.local() 
_DOWNLOAD_LOCK = threading.Lock() # 保护“下载”
_LOAD_LOCK = threading.Lock()     # 添加一个全局锁用于“加载”

# 预下载函数
def pre_download_whisper_model(model_name: str):
    """
    (Main-Thread) 确保 Whisper 模型在多线程转录开始前已被下载。
    这是为了防止多个线程同时尝试下载同一个模型文件。
    """
    with _DOWNLOAD_LOCK:
        print(f"  > 正在检查并预下载 Whisper 模型: {model_name}...")
        try:
            # load_model 会检查缓存，如果不存在则下载
            # 我们加载一次以触发下载，然后丢弃模型实例
            _ = whisper.load_model(model_name) 
            print(f"  > ✅ 模型 '{model_name}' 已准备就绪 (已缓存或已下载)。")
            return True
        except Exception as e:
            print(f"  > 严重错误：无法下载/加载 Whisper 模型 '{model_name}'。")
            print(f"  > 详细错误: {e}")
            raise # 重新抛出异常，以便 main.py 捕获并停止处理

def _load_whisper_model(model_name: str):
    # 这个函数现在加载模型并返回它
    print(f"  > 正在为当前线程加载本地 Whisper 模型: {model_name}...")
    try:
        # (修改) 添加加载锁
        # 确保一次只有一个线程在执行 whisper.load_model()
        # 这可以防止因同时从磁盘读取/反序列化模型文件而引起的竞争条件
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

# (修改) 接受 model_name 参数
def transcribe_single_audio_chunk(audio_path: str, model_name: str) -> str | None:
    """调用本地 Whisper 模型转录单个音频文件 (线程安全)"""
    
    # (修改) 检查当前线程是否已加载 *正确* 的模型
    model_loaded_correctly = (
        hasattr(MODEL_STORAGE, "model") and 
        hasattr(MODEL_STORAGE, "model_name") and
        MODEL_STORAGE.model_name == model_name
    )

    if not model_loaded_correctly:
        try:
            # (修改) 加载指定的模型
            # 这里的 _load_whisper_model 现在是线程安全的了
            MODEL_STORAGE.model = _load_whisper_model(model_name)
            MODEL_STORAGE.model_name = model_name # (修改) 存储当前加载的模型名称
        except Exception:
            return None # 如果模型加载失败，则无法转录

    model = MODEL_STORAGE.model

    audio_filename = os.path.basename(audio_path)
    # (修改) 打印正确的模型名称
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
        # (修改) 打印正确的模型名称
        print(f"  > [Whisper] 本地转录失败 (文件: {audio_filename}, 模型: {model_name}): {e}")
        return None

QWEN_RETRY_EXCEPTIONS = (
    subprocess.CalledProcessError, # 以防 SDK 内部调用 ffmpeg (虽然不太可能)
    ConnectionError,
    Exception # 捕获 DashScope SDK 的通用 API 错误
)

@retry(max_retries=3, delay=5, allowed_exceptions=QWEN_RETRY_EXCEPTIONS)
def transcribe_with_qwen(audio_path: str) -> str | None:
    """
    调用 Qwen ASR API (qwen3-asr-flash) 转录单个音频文件。
    使用 file:// 协议上传本地文件。
    """
    audio_filename = os.path.basename(audio_path)
    print(f"  > [Qwen API] 正在提交: {audio_filename} (线程: {threading.current_thread().name})")

    try:
        # 确保 API Key 已设置
        if not DASHSCOPE_API_KEY:
            print("  > 错误: DASHSCOPE_API_KEY 未在环境中设置。")
            return None
        
        # 确保文件存在
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        # 获取绝对路径，Qwen API 需要 file:// 协议和绝对路径
        abs_audio_path = os.path.abspath(audio_path)
        
        # Qwen3-ASR API 不使用 system prompt
        messages = [
            {
                "role": "user",
                "content": [
                    # DashScope SDK 要求本地文件使用 file:// 协议
                    {"audio": f"file://{abs_audio_path}"}, 
                ]
            }
        ]

        # 设置北京地域的 API URL (如果需要新加坡，请更改)
        dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

        # (修改) 直接调用 dashscope.MultiModalConversation
        response = dashscope.MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model="qwen3-asr-flash", # 使用文档中指定的模型
            messages=messages,
            result_format="message", # 获取包含 message 的完整响应
            asr_options={
                "enable_itn": True # 开启逆文本规范化 (例如 "一百" -> "100")
            }
        )
        # --- 结束修改 ---

        # 解析响应
        if response.status_code == 200 and response.output:
            content_list = response.output.choices[0].message.content
            # 查找第一个 text 类型的 content
            for item in content_list:
                if "text" in item:
                    transcription = item["text"]
                    print(f"  > ✅ [Qwen API] 文件 '{audio_filename}' 转录成功！")
                    return transcription
            
            # 如果循环结束没找到 text
            print(f"  > 错误 [Qwen API]: API 响应成功，但在 content 中未找到 'text'。 响应: {response}")
            return None
        else:
            # API 返回了非 200 状态码
            print(f"  > 错误 [Qwen API]: API 调用失败 (文件: {audio_filename})。")
            print(f"  > 状态码: {response.status_code}, 响应: {response}")
            # 抛出异常以触发 retry
            raise Exception(f"Qwen ASR API 错误: {response.status_code} - {response.message}")

    except FileNotFoundError:
        print(f"  > 错误 [Qwen API]: 找不到音频文件: {audio_path}")
        return None
    except Exception as e:
        # 捕获所有其他错误 (包括 retry 失败后的最终错误)
        print(f"  > [Qwen API] 转录失败 (文件: {audio_filename}): {e}")
        # 如果 retry 装饰器在，这个异常会被它捕获并重试
        # 如果重试次数用尽，这个 raise 会把异常抛给 ThreadPoolExecutor
        raise e 
    
# if __name__ == '__main__':
#     # ... (测试代码保持不变) ...
#     # 测试时需要提供模型名称
#     pre_download_whisper_model("base")
#     transcription = transcribe_with_qwen("chunk_003.mp3")
#     print(f"\n转录结果:\n{transcription}")