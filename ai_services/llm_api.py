import requests
import json
from typing import Generator, Union
from core.config import LLM_CONFIG 
from ai_services.prompts import PROMPT_NOTES_STEM, PROMPT_NOTES_HASS, PROMPT_NOTES_REFINER, PROMPT_NOTES_MEDICAL
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
        except Exception:
            pass

@retry(max_retries=3, delay=5, allowed_exceptions=(requests.exceptions.RequestException,))
def _call_llm_api(messages: list, stream_output: bool) -> Union[Generator[str, None, None], str]:
    if not LLM_CONFIG.get('api_key'):
        raise ValueError("API Key 未在 LLM_CONFIG 中配置。")
        
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
    
    response = requests.post(api_url, json=payload, headers=headers, timeout=600, stream=stream_output)
    response.raise_for_status() 

    if stream_output:
        return _handle_streaming_response(response)
    else:
        result = response.json()

        if 'choices' in result and result['choices'][0]['message']['content']:
            return result['choices'][0]['message']['content']
        
        error_msg = result.get('error') or result.get('error_msg') or "API 响应异常，未包含有效内容"
        raise Exception(f"API 错误: {error_msg}")

def run_llm_generation(input_text: str, stream_output: bool, note_type: str, additional_instructions: str = "") -> Union[Generator[str, None, None], str]:
    if note_type == "STEM":
        prompt_template = PROMPT_NOTES_STEM
    elif note_type == "Medical":
        prompt_template = PROMPT_NOTES_MEDICAL
    else: 
        prompt_template = PROMPT_NOTES_HASS

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