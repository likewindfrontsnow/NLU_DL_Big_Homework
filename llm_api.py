# llm_api.py
import requests
import json
import time
from config import LLM_CONFIG 
from prompts import PROMPT_NOTES_STEM
from utils import retry # 导入 retry 装饰器

@retry(max_retries=3, delay=5, allowed_exceptions=(requests.exceptions.RequestException,))
def run_llm_generation(input_text: str, query: str) -> str:
    """
    (非流式版本) 运行 LLM 生成。
    - 包含重试逻辑。
    - 成功则返回完整的 LLM 响应文本。
    - 失败则（在重试后）抛出异常。
    """
    
    # --- 步骤 1: 根据 query 选择 Prompt ---
    if query == "Notes":
        final_prompt = PROMPT_NOTES_STEM.format(source_transcript=input_text)
    else:
        raise ValueError(f"功能暂未实现: 仅支持 'Notes' 模式。您请求的是 '{query}'。")
    
    # --- 步骤 2: 准备 API 请求 ---
    api_url = LLM_CONFIG["base_url"] + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_CONFIG['api_key']}", 
        "Content-Type": "application/json"
    }
    
    messages = [
        {"role": "system", "content": final_prompt} 
    ]
    
    payload = {
        "model": LLM_CONFIG["model"], 
        "messages": messages,
        "stream": False
    }
    
    print(f"正在连接到 {LLM_CONFIG.get('provider_name', 'LLM')} API (模型: {LLM_CONFIG['model']})...")
    
    # --- 步骤 3: 发送请求 (retry 装饰器将处理网络异常) ---
    response = requests.post(api_url, json=payload, headers=headers, timeout=600)

    response.raise_for_status()

    # --- 步骤 4: 解析结果 ---
    result = response.json()

    if 'choices' in result and result['choices'][0]['message']['content']:
        print(f"✅ {LLM_CONFIG.get('provider_name', 'LLM')} API 响应成功。")
        full_text = result['choices'][0]['message']['content']
        
        if "</think>" in full_text:
            full_text = full_text.split("</think>")[-1].strip()
            
        return full_text
    elif 'error' in result: 
        print(f"❌ {LLM_CONFIG.get('provider_name', 'LLM')} API 返回业务错误: {result.get('error')}")
        raise Exception(f"API 错误: {result.get('error')}")
    elif 'error_msg' in result: 
        print(f"❌ API 返回业务错误: {result.get('error_msg')}")
        raise Exception(f"API 错误: {result.get('error_msg')}")
    else:
        print(f"❌ API 响应结构异常: {result}")
        raise Exception("API 响应异常，未包含有效内容")