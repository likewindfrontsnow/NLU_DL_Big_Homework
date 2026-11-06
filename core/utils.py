import functools
import threading
import requests
from core.tasks import check_cancelled


class APIError(RuntimeError):
    def __init__(self, status, code, message):
        self.status_code = int(status or 0)
        self.code = str(code or '')
        super().__init__(f'API 错误 ({self.status_code}, {self.code}): {message}')


def retryable(error):
    if isinstance(error, (requests.Timeout, requests.ConnectionError)):
        return True
    response = getattr(error, 'response', None)
    status = getattr(error, 'status_code', None)
    if response is not None:
        status = response.status_code
    code = getattr(error, 'code', '').lower()
    if any(word in code for word in ('arrearage', 'allocationquota', 'quotaexhausted', 'insufficient')):
        return False
    return status in (408, 429, 500, 502, 503, 504)


def retry(max_retries=3, delay=2, backoff_factor=2, allowed_exceptions=()):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cancel_event = kwargs.get('cancel_event')
            for attempt in range(max_retries):
                check_cancelled(cancel_event)
                try:
                    result = func(*args, **kwargs)
                    check_cancelled(cancel_event)
                    return result
                except Exception as error:
                    if (attempt + 1 >= max_retries or not retryable(error)
                            or (allowed_exceptions and not isinstance(error, allowed_exceptions))):
                        raise
                    (cancel_event or threading.Event()).wait(delay * backoff_factor ** attempt)
                    check_cancelled(cancel_event)
        return wrapper
    return decorator
