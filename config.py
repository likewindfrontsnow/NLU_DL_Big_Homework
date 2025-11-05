# config.py
import os
from dotenv import load_dotenv

load_dotenv(override=True)

LLM_PROVIDER_NAME = os.getenv("LLM_PROVIDER_NAME", "LLM")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not LLM_BASE_URL:
    raise ValueError("错误：请在 .env 文件中设置您的 LLM_BASE_URL")
if not LLM_MODEL:
    raise ValueError("错误：请在 .env 文件中设置您的 LLM_MODEL")


LLM_CONFIG = {
    "provider_name": LLM_PROVIDER_NAME,
    "base_url": LLM_BASE_URL,
    "api_key": LLM_API_KEY,
    "model": LLM_MODEL
}