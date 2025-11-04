# llm_api.py
import requests
import json
import time
from config import ERNIE_CONFIG
from prompts import PROMPT_NOTES_STEM
from utils import retry # 导入 retry 装饰器

@retry(max_retries=3, delay=5, allowed_exceptions=(requests.exceptions.RequestException,))
def run_ernie_generation(input_text: str, query: str) -> str:
    """
    (非流式版本) 运行 ERNIE LLM 生成。
    - 包含重试逻辑。
    - 成功则返回完整的 LLM 响应文本。
    - 失败则（在重试后）抛出异常。
    """
    
    # --- 步骤 1: 根据 query 选择 Prompt ---
    # 目前我们只实现了 "Notes" 功能
    if query == "Notes":
        final_prompt = PROMPT_NOTES_STEM.format(source_transcript=input_text)
    else:
        # 如果是 Q&A 或 Quiz，暂时抛出错误，我们将在后续步骤中实现
        raise ValueError(f"功能暂未实现: 仅支持 'Notes' 模式。您请求的是 '{query}'。")
    
    # --- 步骤 2: 准备 API 请求 ---
    api_url = ERNIE_CONFIG["base_url"] + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {ERNIE_CONFIG['api_key']}",
        "Content-Type": "application/json"
    }
    
    messages = [
        # 使用 'system' 角色来承载复杂的 Prompt
        {"role": "system", "content": final_prompt} 
    ]
    
    payload = {
        "model": ERNIE_CONFIG["model"],
        "messages": messages,
        "stream": False # 根据您的示例，我们使用非流式
    }
    
    print(f"正在连接到 ERNIE API (模型: {ERNIE_CONFIG['model']})...")
    
    # --- 步骤 3: 发送请求 (retry 装饰器将处理网络异常) ---
    # 为 LLM 生成设置一个较长的超时时间
    response = requests.post(api_url, json=payload, headers=headers, timeout=600)

    # 检查 HTTP 错误 (例如 401, 404, 500)
    response.raise_for_status()

    # --- 步骤 4: 解析结果 ---
    result = response.json()

    if 'choices' in result and result['choices'][0]['message']['content']:
        print("✅ ERNIE API 响应成功。")
        full_text = result['choices'][0]['message']['content']
        
        # Dify 工作流的 LLM_NOTE_REVIEWER1 节点 似乎做了后处理
        # 我们在这里模拟一下，移除 <think> 标签（如果存在）
        if "</think>" in full_text:
            full_text = full_text.split("</think>")[-1].strip()
            
        return full_text
    elif 'error_msg' in result:
        print(f"❌ ERNIE API 返回业务错误: {result.get('error_msg')}")
        raise Exception(f"ERNIE API 错误: {result.get('error_msg')}")
    else:
        print(f"❌ API 响应结构异常: {result}")
        raise Exception("API 响应异常，未包含有效内容")