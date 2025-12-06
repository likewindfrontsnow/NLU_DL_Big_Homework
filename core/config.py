import os
import json
from dotenv import load_dotenv

load_dotenv(override=True)

LLM_PROVIDER_NAME = os.getenv("LLM_PROVIDER_NAME", "LLM")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

DEFAULT_MODELS = {
    "llm_models": [
        "qwen3-max",
        "qwen-plus-2025-07-28",
        "qwen-flash-2025-07-28",
        "deepseek-v3.2-exp",
        "qwen-flash",
    ],
    "vlm_models": [
        "qwen3-vl-plus",
        "qwen3-vl-flash",
        "qwen3-vl-30b-a3b-thinking",
        "qwen-vl-max",
        "qwen-vl-plus",
        "qvq-max",
        "qvq-plus",
        "qvq-72b-preview",
        "qwen-vl-ocr-latest"
    ],
    "asr_models": [
        "qwen3-omni-30b-a3b-captioner",
        "qwen3-asr-flash",
        "qwen-audio-turbo-latest",
        "qwen-audio-asr-latest",
    ]
}

MODELS_CONFIG_PATH = "models.json"

def load_models_config():
    if os.path.exists(MODELS_CONFIG_PATH):
        try:
            with open(MODELS_CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                return {
                    "llm_models": config.get("llm_models", DEFAULT_MODELS["llm_models"]),
                    "vlm_models": config.get("vlm_models", DEFAULT_MODELS["vlm_models"]),
                    "asr_models": config.get("asr_models", DEFAULT_MODELS["asr_models"]),
                }
        except Exception:
            return DEFAULT_MODELS
    return DEFAULT_MODELS

def save_models_config(config):
    try:
        with open(MODELS_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving model config: {e}")

current_config = load_models_config()

SUPPORTED_LLM = current_config["llm_models"]
SUPPORTED_VLM = current_config["vlm_models"]
SUPPORTED_ASR_MODELS = current_config["asr_models"]

SUPPORTED_IV_MODELS = [
    "qwen-image-edit-plus",
    "qwen-image-plus",
    "wan2.5-i2v-preview",
    "wan2.5-t2i-preview",
    "wan2.5-t2v-preview",
]

SUPPORTED_TTS_MODELS = [
    "qwen3-tts-flash",
]

ASR_MODELS_WITH_CONTEXT_SUPPORT = [
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
    "asr_backup_model": "qwen-audio-asr-latest",
    "vlm_backup_model": "qwen-vl-plus",
}