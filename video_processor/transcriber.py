# transcriber.py
import os
import whisper
import threading

MODEL_STORAGE = threading.local() 

MODEL_NAME = "tiny" 

def _load_whisper_model(model_name: str):
    # 这个函数现在加载模型并返回它
    print(f"  > 正在为当前线程加载本地 Whisper 模型: {model_name}...")
    try:
        model = whisper.load_model(model_name) 
        print(f"  > ✅ 本地模型 '{model_name}' (线程 {threading.current_thread().name}) 加载成功！")
        return model
    except Exception as e:
        print(f"  > 错误：无法加载本地 Whisper 模型 '{model_name}'。请确保已正确安装 whisper 库及其依赖。")
        print(f"  > 详细错误: {e}")
        raise

def transcribe_single_audio_chunk(audio_path: str) -> str | None:
    """调用本地 Whisper 模型转录单个音频文件 (线程安全)"""
    
    if not hasattr(MODEL_STORAGE, "model"):
        try:
            MODEL_STORAGE.model = _load_whisper_model(MODEL_NAME)
        except Exception:
            return None # 如果模型加载失败，则无法转录

    model = MODEL_STORAGE.model

    audio_filename = os.path.basename(audio_path)
    print(f"  > 正在转录: {audio_filename} (使用本地模型: {MODEL_NAME}, 线程: {threading.current_thread().name})")
    
    try:
        if not os.path.exists(audio_path):
             raise FileNotFoundError
        
        result = model.transcribe(audio_path)
        transcription = result["text"]

        print(f"  > ✅ 文件 '{audio_filename}' 转录成功！")
        return transcription
    
    except FileNotFoundError:
        print(f"  > 错误：找不到音频文件: {audio_path}")
        return None 
    except Exception as e:
        print(f"  > 本地转录失败 (文件: {audio_filename}): {e}")
        return None
    
# if __name__ == '__main__':
#     transcription = transcribe_single_audio_chunk("chunk_002.mp3")
#     print(f"\n转录结果:\n{transcription}")
