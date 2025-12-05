# llm_api.py
import requests
import json
import time
from typing import Generator, Union
from core.config import LLM_CONFIG 
from prompts import PROMPT_NOTES_STEM, PROMPT_NOTES_HASS, PROMPT_NOTES_REFINER,PROMPT_NOTES_MEDICAL
from core.utils import retry 

def _handle_streaming_response(response: requests.Response) -> Generator[str, None, None]:
    for line in response.iter_lines():
        if not line:
            continue
        line_str = line.decode('utf-8')
        if line_str.startswith("data: "):
            line_str = line_str[6:]
        if line_str == "[DONE]":
            break
        try:
            chunk_data = json.loads(line_str)
            content = ""
            if 'choices' in chunk_data and \
               chunk_data['choices'][0].get('delta', {}).get('content') is not None:
                content = chunk_data['choices'][0]['delta']['content']
            if not content:
                continue
            yield content
        except json.JSONDecodeError:
            print(f"  > (stream) 无法解析行: {line_str}")
        except Exception as e:
            print(f"  > (stream) 处理块时出错: {e}, 块: {line_str}")

@retry(max_retries=3, delay=5, allowed_exceptions=(requests.exceptions.RequestException,))
def _call_llm_api(messages: list, stream_output: bool) -> Union[Generator[str, None, None], str]:
    if not LLM_CONFIG.get('api_key'):
        raise ValueError("API Key 未在 LLM_CONFIG 中配置。请检查 .env 文件或重启应用以输入 Key。")
        
    api_url = LLM_CONFIG["base_url"] + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_CONFIG['api_key']}", 
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": LLM_CONFIG["model"], 
        "messages": messages,
        "stream": stream_output 
    }
    
    print(f"正在连接到 {LLM_CONFIG.get('provider_name', 'LLM')} API (模型: {LLM_CONFIG['model']}, 流式: {stream_output})...")
    
    response = requests.post(api_url, json=payload, headers=headers, timeout=600, stream=stream_output)
    response.raise_for_status() 

    if stream_output:
        print(f"✅ {LLM_CONFIG.get('provider_name', 'LLM')} API 流式响应开始...")
        return _handle_streaming_response(response)
    else:
        print(f"✅ {LLM_CONFIG.get('provider_name', 'LLM')} API 非流式响应成功。")
        result = response.json()

        if 'choices' in result and result['choices'][0]['message']['content']:
            full_text = result['choices'][0]['message']['content']
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

def run_llm_generation(input_text: str, stream_output: bool, note_type: str, additional_instructions: str = "") -> Union[Generator[str, None, None], str]:
    
    if note_type == "STEM":
        prompt_template = PROMPT_NOTES_STEM
        print("  > 使用 STEM (理工科) Prompt")
    elif note_type == "Medical":
        prompt_template = PROMPT_NOTES_MEDICAL
        print("  > 使用 Medical (医学) Prompt")
    else: 
        prompt_template = PROMPT_NOTES_HASS
        print("  > 使用 HASS (人文社科) Prompt")

    if additional_instructions:
        prompt_template += f"\n\n# 用户额外特别指令 (User Custom Instructions)\n请在生成笔记时，严格遵守以下用户提出的额外要求：\n{additional_instructions}"
    final_prompt = prompt_template.format(source_transcript=input_text)
    
    messages = [
        {"role": "system", "content": final_prompt} 
    ]
    
    return _call_llm_api(messages, stream_output)

def refine_llm_generation(original_transcript: str, current_notes: str, user_feedback: str, stream_output: bool) -> Union[Generator[str, None, None], str]:
    messages = [
        {"role": "system", "content": PROMPT_NOTES_REFINER},
        {"role": "user", "content": f"""
        【原始转录稿】
        {original_transcript}
        
        ---
        
        【当前笔记】
        {current_notes}
        
        ---
        
        【用户指令】
        {user_feedback}
        """}
    ]
    
    return _call_llm_api(messages, stream_output)