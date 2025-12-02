# model_verifier.py
import os
import json
import requests
import dashscope
import tempfile
import wave
import struct
from http import HTTPStatus
from config import LLM_CONFIG, SUPPORTED_LLM, SUPPORTED_ASR_MODELS

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
    except Exception as e:
        print(f"创建测试音频失败: {e}")
        if os.path.exists(temp.name):
            os.remove(temp.name)
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
            err_code = err_data.get('error', {}).get('code') or err_data.get('code')
        except:
            err_msg = response.text
            err_code = response.status_code

        if response.status_code == 401:
            return False, f"❌ 认证失败 (401): API Key 无效或过期。详情: {err_msg}"
        elif response.status_code == 404:
            return False, f"❌ 模型不存在 (404): 服务端找不到模型 '{model_name}'，请检查拼写。详情: {err_msg}"
        elif response.status_code == 429:
            return False, f"❌ 请求过多或欠费 (429): 触发限流或余额不足。详情: {err_msg}"
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

    try:
        dashscope.api_key = api_key
        abs_audio_path = os.path.abspath(dummy_audio_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"audio": f"file://{abs_audio_path}"}
                ]
            }
        ]

        response = dashscope.MultiModalConversation.call(
            model=model_name,
            messages=messages,
        )

        if response.status_code == HTTPStatus.OK:
            return True, "✅ 验证通过：模型可用且 Key 有效。"
        
        else:
            code = response.code
            message = response.message
            
            if "InvalidApiKey" in code:
                return False, f"❌ 认证失败: API Key 无效。({code})"
            elif "ModelNotFound" in code or "InvalidModel" in code:
                return False, f"❌ 模型不存在: 无法找到模型 '{model_name}'。({code})"
            elif "Arrears" in code:
                return False, f"❌ 账户欠费: 请检查阿里云账户余额。({code})"
            else:
                return False, f"❌ 调用失败: {message} ({code})"

    except Exception as e:
        return False, f"❌ SDK 调用发生异常: {str(e)}"
    
    finally:
        if os.path.exists(dummy_audio_path):
            try:
                os.remove(dummy_audio_path)
            except:
                pass

def run_full_check():
    """
    读取 config.py 中的配置，对当前选中的 LLM 和 ASR 进行验证。
    """
    print("="*60)
    print("🤖 模型可用性深度验证程序 (Model Verifier)")
    print("="*60)

    current_api_key = LLM_CONFIG.get("api_key")
    if not current_api_key:
        current_api_key = os.getenv("DASHSCOPE_API_KEY")
    
    current_base_url = LLM_CONFIG.get("base_url")
    current_llm = LLM_CONFIG.get("model")
    
    target_asr = "qwen3-asr-flash" 

    if not current_api_key:
        print("❌ 错误：未在 config.py 或环境变量中找到 API Key。请先配置 .env 文件。")
        return

    print(f"\n[1/2] 正在验证当前配置的 LLM: {current_llm}")
    success_llm, msg_llm = verify_llm_model(current_api_key, current_base_url, current_llm)
    print(msg_llm)

    print(f"\n[2/2] 正在验证 ASR 模型: {target_asr}")
    if "qwen" not in target_asr.lower() and "paraformer" not in target_asr.lower():
        print("⚠️ 跳过 ASR 验证：当前验证器仅支持 Qwen/DashScope 系列 ASR 模型。")
    else:
        success_asr, msg_asr = verify_asr_model(current_api_key, target_asr)
        print(msg_asr)

    print("\n" + "="*60)
    if success_llm and (success_asr if 'success_asr' in locals() else True):
        print("🎉 所有检查项通过！系统状态良好。")
    else:
        print("⚠️ 发现潜在问题，请根据上方错误提示进行修正。")
    print("="*60)

if __name__ == "__main__":
    run_full_check()