# config.py
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv(override=True)

ERNIE_API_KEY = os.getenv("ERNIE_API_KEY")

ERNIE_CONFIG = {
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_key": ERNIE_API_KEY,
    "model": "ernie-3.5-8k" 
}

if not ERNIE_API_KEY:
    raise ValueError("错误：请在 .env 文件中设置您的 ERNIE_API_KEY")
