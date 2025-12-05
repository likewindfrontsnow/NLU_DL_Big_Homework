import os
import requests
import dashscope
from dashscope.audio.asr import Transcription
import tempfile
import wave
import struct
from http import HTTPStatus

def _create_dummy_wav(duration_sec=0.5):
    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        with wave.open(temp.name, 'w') as wav_file:
            n_channels = 1
            sampwidth = 2
            framerate = 16000
            n_frames = int(framerate * duration_sec)
            
            wav_file.setparams((n_channels, sampwidth, framerate, n_frames, 'NONE', 'not compressed'))
            for _ in range(n_frames):
                wav_file.writeframes(struct.pack('h', 0))
        return temp.name
    except Exception:
        if os.path.exists(temp.name):
            try:
                os.remove(temp.name)
            except:
                pass
        return None

def verify_llm_model(api_key: str, base_url: str, model_name: str):
    print(f"🔍 [LLM 验证] 正在检查模型: {model_name} ...")
    
    if not api_key or not base_url:
        return False, "缺少 API Key 或 Base URL 配置"

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1 
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            return True, "✅ 验证通过：模型可用且 Key 有效。"
        
        try:
            err_data = response.json()
            err_msg = err_data.get('error', {}).get('message') or err_data.get('message') or str(err_data)
        except:
            err_msg = response.text

        if response.status_code == 401:
            return False, f"❌ 认证失败 (401): API Key 无效或过期。详情: {err_msg}"
        elif response.status_code == 404:
            return False, f"❌ 模型不存在 (404): 服务端找不到模型 '{model_name}'。详情: {err_msg}"
        elif response.status_code == 429:
            return False, f"❌ 请求过多或欠费 (429)。详情: {err_msg}"
        else:
            return False, f"❌ 未知错误 ({response.status_code}): {err_msg}"

    except requests.exceptions.Timeout:
        return False, "❌ 请求超时: 无法连接到服务器，请检查网络。"
    except requests.exceptions.RequestException as e:
        return False, f"❌ 连接异常: {e}"

def verify_asr_model(api_key: str, model_name: str):
    print(f"🔍 [ASR 验证] 正在检查模型: {model_name} ...")
    
    if not api_key:
        return False, "缺少 DashScope API Key"

    dummy_audio_path = _create_dummy_wav()
    if not dummy_audio_path:
        return False, "本地环境错误: 无法生成测试音频文件"

    abs_audio_path = os.path.abspath(dummy_audio_path)
    dashscope.api_key = api_key
    
    is_async_model = "filetrans" in model_name or "fun-asr" in model_name

    try:
        if is_async_model:
            response = Transcription.async_call(
                model=model_name,
                file_urls=[f"file://{abs_audio_path}"],
            )
            
            if response.status_code == HTTPStatus.OK:
                return True, "✅ 验证通过：模型可用 (异步提交成功)。"
            else:
                return False, f"❌ 调用失败: {response.message} ({response.code})"
                
        else:
            messages = [
                {
                    "role": "user",
                    "content": [{"audio": f"file://{abs_audio_path}"}]
                }
            ]
            response = dashscope.MultiModalConversation.call(
                model=model_name,
                messages=messages,
            )

            if response.status_code == HTTPStatus.OK:
                return True, "✅ 验证通过：模型可用。"
            else:
                return False, f"❌ 调用失败: {response.message} ({response.code})"

    except Exception as e:
        return False, f"❌ SDK 调用发生异常: {str(e)}"
    
    finally:
        if os.path.exists(dummy_audio_path):
            try:
                os.remove(dummy_audio_path)
            except:
                pass