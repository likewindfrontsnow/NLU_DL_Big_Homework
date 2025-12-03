import os
from dotenv import load_dotenv

load_dotenv(override=True)

LLM_PROVIDER_NAME = os.getenv("LLM_PROVIDER_NAME", "LLM")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

SUPPORTED_LLM=[
    "qwen3-max",
    "qwen-plus-2025-07-28",
    "qwen-flash-2025-07-28",
    "deepseek-v3.2-exp",
    "qwen-flash",
]

SUPPORTED_VLM=[
    "qwen3-vl-plus",
]

SUPPORTED_IV_MODELS = [
    "qwen-image-edit-plus",
    "qwen-image-plus",
    "wan2.5-i2v-preview",
    "wan2.5-t2i-preview",
    "wan2.5-t2v-preview",
]

SUPPORTED_TTS_MODELS=[
    "qwen3-tts-flash",
]

SUPPORTED_ASR_MODELS = [
    "qwen3-omni-30b-a3b-captioner",
    "qwen3-asr-flash-filetrans",
    "qwen3-asr-flash",   
    "fun-asr-mtl",
    "qwen-audio-turbo-latest",
    "qwen-audio-asr-latest",    
    "paraformer-8k-v1",
    "paraformer-8k-v2",
    "paraformer-v2",
    "paraformer-v1",
    "paraformer-mtl-v1",
    "qwen3-asr-flash-realtime",
    "fun-asr-realtime",
    "paraformer-realtime-8k-v2",
    "paraformer-realtime-v2",
    "paraformer-realtime-8k-v1",
    "paraformer-realtime-v1",
]

ASR_MODELS_WITH_CONTEXT_SUPPORT=[
    "qwen3-asr-flash-filetrans",
    "qwen3-asr-flash",   

]

LLM_CONFIG = {
    "provider_name": LLM_PROVIDER_NAME,
    "base_url": LLM_BASE_URL,
    "api_key": LLM_API_KEY,
    "model": LLM_MODEL,
    "supported_llm": SUPPORTED_LLM,
    "supported_vlm": SUPPORTED_VLM,
    "supported_iv_models": SUPPORTED_IV_MODELS,
    "supported_tts_models": SUPPORTED_TTS_MODELS,
    "supported_asr_models": SUPPORTED_ASR_MODELS,
}