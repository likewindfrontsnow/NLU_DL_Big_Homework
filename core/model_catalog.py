"""百炼目录同步：https://help.aliyun.com/zh/model-studio/list-models。"""
import hashlib
import json
import re
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import requests

CACHE_SECONDS = 24 * 60 * 60
RETRY_SECONDS = 5 * 60
GROUPS = ('llm_models', 'vlm_models', 'asr_models')


def catalog_endpoint(base_url):
    parsed = urlsplit(base_url or '')
    host = parsed.hostname or ''
    legacy = {'dashscope.aliyuncs.com', 'dashscope-intl.aliyuncs.com',
              'dashscope-us.aliyuncs.com', 'cn-hongkong.dashscope.aliyuncs.com'}
    if (parsed.scheme != 'https' or parsed.username or parsed.password or parsed.port
            or not (host in legacy or host.endswith('.maas.aliyuncs.com'))
            or host.startswith(('token-plan.', 'trial.'))):
        raise ValueError('自动同步仅支持百炼标准 API 地址；请使用对应地域/业务空间的按量付费服务地址。当前仍可使用本地列表。')
    return f'https://{host}/api/v1/models'


def catalog_identity(api_key, endpoint):
    return hashlib.sha256(f'{endpoint}\n{api_key}'.encode()).hexdigest()


def fetch_catalog(api_key, endpoint):
    """只在全部分页完整成功时返回结果，不把部分结果当成完整列表。"""
    models = {}
    total = None
    deadline = time.monotonic() + 30
    for page in range(1, 101):
        if time.monotonic() > deadline:
            raise ValueError('目录同步超时，保留原列表')
        with requests.get(endpoint, headers={'Authorization': f'Bearer {api_key}'},
                          params={'page_no': page, 'page_size': 100, 'supports': 'inference'},
                          timeout=(5, 15), allow_redirects=False) as response:
            if response.status_code != 200:
                labels = {401: 'API Key 认证失败', 403: '无权查询该地域或业务空间的模型目录',
                          404: '该地址不支持模型目录接口，请检查地域和业务空间配置', 429: '目录查询被限流'}
                raise ValueError(labels.get(response.status_code, f'目录服务返回 HTTP {response.status_code}'))
            payload = response.json()
        if not isinstance(payload, dict) or payload.get('success') is False:
            raise ValueError('模型目录返回失败，保留原列表')
        output = payload.get('output')
        if not isinstance(output, dict) or not isinstance(output.get('models'), list):
            raise ValueError('模型目录返回格式不正确，保留原列表')
        count = output.get('total')
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError('模型目录为空或缺少总数，保留原列表')
        if total is not None and count != total:
            raise ValueError('分页期间模型目录发生变化，请重新刷新')
        total = count
        previous = len(models)
        for item in output['models']:
            if not isinstance(item, dict):
                raise ValueError('模型目录包含无效记录')
            name = item.get('model') or item.get('id')
            if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:/-]{0,199}', name):
                raise ValueError('模型目录包含无效模型名称')
            models[name] = dict(item, model=name)
        if len(models) == total:
            return list(models.values())
        if len(models) <= previous or len(models) > total:
            raise ValueError('模型分页不完整，保留原列表')
    raise ValueError('模型目录分页超过安全上限，保留原列表')


def offline_dates(value):
    dates = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'offline_time' and isinstance(item, str):
                try:
                    date = datetime.fromisoformat(item.replace('Z', '+00:00'))
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone(timedelta(hours=8)))
                    dates.append(date)
                except ValueError:
                    pass
            elif isinstance(item, (dict, list)):
                dates.extend(offline_dates(item))
    elif isinstance(value, list):
        for item in value:
            dates.extend(offline_dates(item))
    return dates


def group_catalog(models, now=None):
    now = now or datetime.now(timezone.utc)
    groups = {key: [] for key in GROUPS}
    notices = {}
    for item in models:
        name = item['model']
        provider = item.get('inference_provider')
        if provider and provider not in ('aliyun-bailian', 'alibaba-cloud-modelstudio'):
            continue
        dates = offline_dates(item.get('inference_offline_info'))
        # 多部署场景可能有不同下线时间，不据单个部署的日期误删模型。
        if dates:
            notices[name] = '下线信息：' + '、'.join(sorted({d.strftime('%Y-%m-%d') for d in dates}))
            direct = item.get('inference_offline_info') or {}
            if isinstance(direct, dict) and direct.get('offline_time') and all(d <= now for d in dates):
                continue
        caps = set(item.get('capabilities') or [])
        if caps & {'TG', 'Reasoning'} and not caps & {'ASR', 'TTS', 'Realtime-ASR', 'Realtime-Omni'}:
            groups['llm_models'].append(name)
        if 'VU' in caps and (name.startswith(('qwen-vl', 'qvq')) or re.match(r'qwen\d[\w.-]*-vl', name)):
            groups['vlm_models'].append(name)
        if 'ASR' in caps and re.fullmatch(r'qwen3-asr-flash(?:-\d{4}-\d{2}-\d{2})?', name):
            groups['asr_models'].append(name)
    if not groups['llm_models']:
        raise ValueError('目录中未找到兼容的文本生成模型，保留原列表')
    return {key: sorted(set(value)) for key, value in groups.items()}, notices


def sync_catalog(state, api_key, base_url, *, force=False, cache_dir='.cache/model_catalog', now=None):
    """状态及缓存按 Key+端点隔离；失败不覆盖缓存、不清空当前列表。"""
    now = time.time() if now is None else now
    if not api_key:
        return
    try:
        endpoint = catalog_endpoint(base_url)
    except ValueError as error:
        state['catalog_message'] = str(error)
        state['catalog_synced_at'] = None
        return
    identity = catalog_identity(api_key, endpoint)
    changed = identity != state.get('catalog_identity')
    if changed:
        state['catalog_identity'] = identity
        state['catalog_last_attempt'] = 0
        state['catalog_applied_at'] = None
        state['catalog_notices'] = {}
        # 不将前一个账户/端点的目录冒充为新账户的缓存。
        state['catalog_synced_at'] = None
        state['catalog_message'] = ''
        fallback = state.get('catalog_local_models', {})
        for key in GROUPS:
            if key in fallback:
                state[key] = list(fallback[key])
    if not force and now - state.get('catalog_last_attempt', 0) < RETRY_SECONDS:
        return
    path = Path(cache_dir) / f'{identity}.json'
    cached = None
    try:
        cached = json.loads(path.read_text(encoding='utf-8'))
        if (not isinstance(cached.get('fetched_at'), (int, float))
                or not isinstance(cached.get('models'), list)):
            cached = None
    except (OSError, ValueError, AttributeError):
        pass
    state['catalog_last_attempt'] = now
    warning = ''
    fetched = False
    if force or cached is None or not 0 <= now - cached['fetched_at'] < CACHE_SECONDS:
        try:
            models = fetch_catalog(api_key, endpoint)
            group_catalog(models)  # 先验证，不能用空分类覆盖成功缓存。
            cached = {'fetched_at': now, 'models': models}
            fetched = True
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(f'.{uuid.uuid4().hex}.tmp')
                try:
                    temporary.write_text(json.dumps(cached, ensure_ascii=False), encoding='utf-8')
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            except OSError:
                warning = '同步成功，但无法写入磁盘缓存。'
        except (requests.RequestException, ValueError, TypeError, KeyError):
            warning = '同步失败：请检查网络、API Key、地域和业务空间地址。已保留上次成功列表或本地列表。'
    if cached is not None:
        try:
            groups, notices = group_catalog(cached['models'])
            if fetched or changed or state.get('catalog_applied_at') != cached['fetched_at']:
                for key, values in groups.items():
                    if values:
                        state[key] = values
                state['catalog_applied_at'] = cached['fetched_at']
            missing = [key.split('_')[0].upper() for key, value in groups.items() if not value]
            if missing:
                warning += ' 未返回兼容的 ' + '/'.join(missing) + ' 模型，这些分类保留本地列表。'
            state['catalog_notices'] = notices
            state['catalog_synced_at'] = cached['fetched_at']
        except (ValueError, TypeError, KeyError, AttributeError):
            warning = '缓存格式无效，已保留本地模型列表，请点击刷新重试。'
    state['catalog_message'] = warning


def model_label(state, name):
    notice = state.get('catalog_notices', {}).get(name)
    return f'{name}（{notice}）' if notice else name
