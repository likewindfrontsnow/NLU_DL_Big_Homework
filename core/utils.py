import time
import functools
import random

def retry(max_retries=3, delay=2, backoff_factor=2, allowed_exceptions=()):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if allowed_exceptions and not isinstance(e, allowed_exceptions):
                        raise

                    attempts += 1
                    if attempts >= max_retries:
                        raise e
                    
                    error_str = str(e).lower()
                    if "429" in error_str or "quota" in error_str:
                        sleep_time = current_delay * (0.5 + random.random())
                    else:
                        sleep_time = current_delay
                    
                    time.sleep(sleep_time)
                    current_delay *= backoff_factor 
        return wrapper
    return decorator