# transcriber.py
import os
import whisper

WHISPER_MODEL = None 
MODEL_NAME = "tiny" 

def _load_whisper_model(model_name: str):
    global WHISPER_MODEL
    print(f"  > 正在加载本地 Whisper 模型: {model_name}...")
    try:
        WHISPER_MODEL = whisper.load_model(model_name) 
        print(f"  > ✅ 本地模型 '{model_name}' 加载成功！")
    except Exception as e:
        print(f"  > 错误：无法加载本地 Whisper 模型 '{model_name}'。请确保已正确安装 whisper 库及其依赖。")
        print(f"  > 详细错误: {e}")
        raise

def transcribe_single_audio_chunk(audio_path: str) -> str | None:
    """调用本地 Whisper 模型转录单个音频文件"""
    
    # 首次调用时加载模型
    if WHISPER_MODEL is None:
        try:
            _load_whisper_model(MODEL_NAME)
        except Exception:
            return None # 如果模型加载失败，则无法转录

    audio_filename = os.path.basename(audio_path)
    print(f"  > 正在转录: {audio_filename} (使用本地模型: {MODEL_NAME})")
    
    try:
        # 1. 检查文件是否存在
        if not os.path.exists(audio_path):
             raise FileNotFoundError
        
        # 2. 调用本地模型的 transcribe 方法
        result = WHISPER_MODEL.transcribe(audio_path)
        transcription = result["text"]

        print(f"  > ✅ 文件 '{audio_filename}' 转录成功！")
        return transcription
    
    except FileNotFoundError:
        print(f"  > 错误：找不到音频文件: {audio_path}")
        return None 
    except Exception as e:
        # 捕获所有其他转录过程中可能出现的错误
        print(f"  > 本地转录失败: {e}")
        # 这里不进行重试，因为本地失败通常是环境或文件问题
        return None
    
# if __name__ == '__main__':
#     transcription = transcribe_single_audio_chunk("chunk_002.mp3")
#     print(f"\n转录结果:\n{transcription}")
#     pass