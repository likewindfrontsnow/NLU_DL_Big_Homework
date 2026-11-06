import requests
import json
from typing import Generator, Union
from core.config import LLM_CONFIG 
from ai_services.prompts import PROMPT_NOTES_STEM, PROMPT_NOTES_HASS, PROMPT_NOTES_REFINER, PROMPT_NOTES_MEDICAL
from core.tasks import check_cancelled, snapshot_config
from core.utils import retry 


def refinement_events(**kwargs):
    result = refine_llm_generation(**kwargs)
    try:
        if kwargs.get('stream_output'):
            for text in result:
                check_cancelled(kwargs.get('cancel_event'))
                yield 'llm_chunk', text
        else:
            check_cancelled(kwargs.get('cancel_event'))
            yield 'llm_chunk', result
    finally:
        close = getattr(result, 'close', None)
        if close:
            close()

def _handle_streaming_response(response: requests.Response, cancel_event=None) -> Generator[str, None, None]:
    finished = False
    try:
        for line in response.iter_lines():
            check_cancelled(cancel_event)
            if not line:
                continue
            line_str = line.decode('utf-8').strip()
            if line_str.startswith((':', 'event:', 'id:', 'retry:')):
                continue
            if line_str.startswith('data:'):
                line_str = line_str[5:].strip()
            if line_str == '[DONE]':
                finished = True
                break
            chunk_data = json.loads(line_str)
            if chunk_data.get('error'):
                raise RuntimeError(f"流式生成失败: {chunk_data['error']}")
            for choice in chunk_data.get('choices', []):
                reason = choice.get('finish_reason')
                if reason is not None:
                    if reason != 'stop':
                        raise RuntimeError(f"笔记生成未完整结束: {reason}")
                    finished = True
                content = choice.get('delta', {}).get('content')
                if content:
                    yield content
        if not finished:
            raise RuntimeError("流式响应意外中断，笔记尚未完整生成。")
    finally:
        response.close()

@retry(max_retries=3, delay=5, allowed_exceptions=(requests.exceptions.RequestException,))
def _call_llm_api(messages: list, stream_output: bool, *, config=None, cancel_event=None) -> Union[Generator[str, None, None], str]:
    config = snapshot_config(LLM_CONFIG if config is None else config)
    check_cancelled(cancel_event)
    if not config.get('api_key') or not config.get('base_url'):
        raise ValueError("缺少 API Key 或 Base URL 配置")
    response = requests.post(
        config['base_url'].rstrip('/') + '/chat/completions',
        json={'model': config['model'], 'messages': messages, 'stream': stream_output},
        headers={'Authorization': f"Bearer {config['api_key']}", 'Content-Type': 'application/json'},
        timeout=(10, 60), stream=stream_output)
    try:
        response.raise_for_status()
        check_cancelled(cancel_event)
        if stream_output:
            return _handle_streaming_response(response, cancel_event)
        result = response.json()
        choices = result.get('choices') or []
        if choices and choices[0].get('message', {}).get('content'):
            if choices[0].get('finish_reason') not in (None, 'stop'):
                raise RuntimeError(f"笔记生成未完整结束: {choices[0]['finish_reason']}")
            return choices[0]['message']['content']
        raise RuntimeError(f"API 响应异常: {result.get('error', '未包含有效内容')}")
    except BaseException:
        response.close()
        raise
    finally:
        if not stream_output:
            response.close()


def run_llm_generation(input_text: str, stream_output: bool, note_type: str, additional_instructions: str = "", reference_material: str = "", *, config=None, cancel_event=None) -> Union[Generator[str, None, None], str]:
    if note_type == "STEM":
        prompt_template = PROMPT_NOTES_STEM
    elif note_type == "Medical":
        prompt_template = PROMPT_NOTES_MEDICAL
    elif note_type == "None":
        prompt_template = None
    else:
        prompt_template = PROMPT_NOTES_HASS

    ref_content = reference_material if reference_material.strip() else "(无参考资料)"

    if prompt_template is None:
        user_content = f"请根据以下内容生成笔记：\n\n{input_text}"
        if reference_material.strip():
            user_content += f"\n\n参考资料：\n{ref_content}"
        if additional_instructions:
            user_content += f"\n\n额外要求：\n{additional_instructions}"
        messages = [{"role": "user", "content": user_content}]
    else:
        final_prompt = prompt_template.format(source_transcript=input_text, reference_material=ref_content)
        if additional_instructions:
            final_prompt += f"\n\n# 用户额外特别指令 (User Custom Instructions)\n请在生成笔记时，严格遵守以下用户提出的额外要求：\n{additional_instructions}"

        messages = [
            {"role": "system", "content": final_prompt}
        ]

    return _call_llm_api(messages, stream_output, config=config, cancel_event=cancel_event)

def refine_llm_generation(original_transcript: str, current_notes: str, user_feedback: str, stream_output: bool, *, config=None, cancel_event=None) -> Union[Generator[str, None, None], str]:
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
    
    return _call_llm_api(messages, stream_output, config=config, cancel_event=cancel_event)
