import base64
import os
import dashscope
from core.config import LLM_CONFIG
from core.tasks import check_cancelled, snapshot_config
from core.utils import APIError, retry, retryable


def _seconds_to_timestamp(seconds):
    return f'{int(seconds // 60):02d}:{int(seconds % 60):02d}'


@retry(max_retries=3, delay=2)
def _execute_transcription(api_key, model_name, audio_path, context, offset, *, cancel_event=None):
    check_cancelled(cancel_event)
    with open(audio_path, 'rb') as source:
        audio = 'data:audio/mpeg;base64,' + base64.b64encode(source.read()).decode('ascii')
    messages = []
    if context:
        messages.append({'role': 'system', 'content': [{'text': context}]})
    messages.append({'role': 'user', 'content': [{'audio': audio}]})
    response = dashscope.MultiModalConversation.call(
        api_key=api_key, model=model_name, messages=messages,
        result_format='message', asr_options={'enable_itn': True}, request_timeout=30)
    check_cancelled(cancel_event)
    if response.status_code != 200:
        raise APIError(response.status_code, response.code, response.message)
    text = ''.join(item['text'] for item in response.output.choices[0].message.content if 'text' in item)
    return f'[{_seconds_to_timestamp(offset)}] {text}\n' if text else ''


def transcribe_with_qwen(audio_path, asr_context=None, model_name='qwen3-asr-flash',
                         start_offset_seconds=0, *, config=None, cancel_event=None):
    config = snapshot_config(LLM_CONFIG if config is None else config)
    check_cancelled(cancel_event)
    key = config.get('asr_api_key') or config.get('api_key')
    if not key:
        raise ValueError('未设置语音转录 API Key')
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(audio_path)
    try:
        return _execute_transcription(key, model_name, audio_path, asr_context or '',
                                      start_offset_seconds, cancel_event=cancel_event)
    except APIError as error:
        backup = config.get('asr_backup_model')
        if not retryable(error) or not backup or backup == model_name:
            raise
        return _execute_transcription(key, backup, audio_path, asr_context or '',
                                      start_offset_seconds, cancel_event=cancel_event)
