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
                    # 如果指定了允许重试的异常，且当前异常不在列表中，则直接抛出
                    if allowed_exceptions and not isinstance(e, allowed_exceptions):
                        raise

                    attempts += 1
                    if attempts >= max_retries:
                        print(f"Function '{func.__name__}' failed after {max_retries} attempts.")
                        raise e
                    
                    # 检查是否为限流错误，如果是，添加一点随机抖动
                    error_str = str(e).lower()
                    if "429" in error_str or "quota" in error_str:
                        sleep_time = current_delay * (0.5 + random.random())
                    else:
                        sleep_time = current_delay
                    
                    print(f"Attempt {attempts}/{max_retries} failed: {e}. Retrying in {sleep_time:.2f}s...")
                    
                    time.sleep(sleep_time)
                    current_delay *= backoff_factor 
        return wrapper
    return decorator