# llm_api.py
import requests
import json
import time
from typing import Generator, Union # 导入类型
from config import LLM_CONFIG 
from prompts import PROMPT_NOTES_STEM
from utils import retry # 导入 retry 装饰器

# --- (修改) 辅助函数：移除了 <think> 标签处理逻辑 ---
def _handle_streaming_response(response: requests.Response) -> Generator[str, None, None]:
    """
    (私有) 处理流式 API 响应的生成器。
    - 兼容 OpenAI 格式 (data: {...})
    """
    # (已移除) 不再需要 buffer 和 think_tag_found

    for line in response.iter_lines():
        if not line:
            continue

        line_str = line.decode('utf-8')
        
        if line_str.startswith("data: "):
            line_str = line_str[6:] # 移除 "data: " 前缀
        
        if line_str == "[DONE]":
            break # 流结束
            
        try:
            chunk_data = json.loads(line_str)
            
            content = ""
            # 检查标准 OpenAI 兼容的流式块
            if 'choices' in chunk_data and \
               chunk_data['choices'][0].get('delta', {}).get('content') is not None:
                content = chunk_data['choices'][0]['delta']['content']
            
            if not content:
                continue # 空块或非内容块

            # --- (已移除) <think> 标签处理逻辑 ---
            # 现在我们直接产出内容
            yield content

        except json.JSONDecodeError:
            print(f"  > (stream) 无法解析行: {line_str}")
        except Exception as e:
            print(f"  > (stream) 处理块时出错: {e}, 块: {line_str}")
# --- 结束修改 ---


@retry(max_retries=3, delay=5, allowed_exceptions=(requests.exceptions.RequestException,))
def run_llm_generation(input_text: str, query: str, stream_output: bool) -> Union[Generator[str, None, None], str]:
    """
    (流式/非流式 兼容版本) 运行 LLM 生成。
    - 包含重试逻辑。
    - stream_output=True: 返回一个文本块生成器。
    - stream_output=False: 返回一个包含完整文本的字符串。
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
        "stream": stream_output # 动态设置 stream 参数
    }
    
    print(f"正在连接到 {LLM_CONFIG.get('provider_name', 'LLM')} API (模型: {LLM_CONFIG['model']}, 流式: {stream_output})...")
    
    # --- 步骤 3: 发送请求 (retry 装饰器将处理网络异常) ---
    response = requests.post(api_url, json=payload, headers=headers, timeout=600, stream=stream_output)

    response.raise_for_status() # 抛出 HTTP 错误 (4xx, 5xx)

    # --- 步骤 4: 根据是否流式处理结果 ---
    if stream_output:
        print(f"✅ {LLM_CONFIG.get('provider_name', 'LLM')} API 流式响应开始...")
        # (修改) 返回更新后的流式处理生成器
        return _handle_streaming_response(response)
    else:
        print(f"✅ {LLM_CONFIG.get('provider_name', 'LLM')} API 非流式响应成功。")
        result = response.json()

        if 'choices' in result and result['choices'][0]['message']['content']:
            full_text = result['choices'][0]['message']['content']
            
            # --- (修改) 移除了 <think> 标签处理逻辑 ---
            # if "</think>" in full_text:
            #     full_text = full_text.split("</think>")[-1].strip()
            
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