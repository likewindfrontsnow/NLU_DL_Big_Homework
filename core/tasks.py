import concurrent.futures
import copy
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path


class TaskCancelled(Exception):
    pass


def check_cancelled(cancel_event=None):
    if cancel_event is not None and cancel_event.is_set():
        raise TaskCancelled('任务已停止')


def validate_output_name(name):
    name = name.strip()
    if (not name or len(name) > 64 or name.endswith(('.', ' '))
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or name.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}):
        raise ValueError('文件名不能为空，不能包含路径、非法字符或 Windows 保留名称，且不能超过 64 个字符。')
    return name


def create_task_paths(name):
    name = validate_output_name(name)
    task_id = uuid.uuid4().hex
    work = Path('temp') / 'tasks' / task_id
    output = Path('output') / f'{name}_{task_id}'
    work.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    return task_id, work, output


def snapshot_config(config):
    return copy.deepcopy(config)


def background_events(events, cancel_event, cleanup=None):
    """后台执行阻塞流程；界面定期获得心跳，可随时发出取消信号。"""
    messages = queue.Queue(maxsize=64)
    done = threading.Event()

    def send(item):
        while not cancel_event.is_set():
            try:
                messages.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def run():
        try:
            for event in events:
                check_cancelled(cancel_event)
                send(('event', event))
        except TaskCancelled:
            pass
        except Exception as error:
            send(('error', error))
        finally:
            try:
                events.close()
            finally:
                try:
                    if cleanup:
                        cleanup()
                finally:
                    done.set()

    threading.Thread(target=run, daemon=True).start()
    try:
        while not done.is_set() or not messages.empty():
            check_cancelled(cancel_event)
            try:
                kind, value = messages.get(timeout=0.1)
            except queue.Empty:
                yield ('heartbeat', None)
                continue
            if kind == 'error':
                raise value
            yield value
    finally:
        cancel_event.set()


def completed_futures(futures, cancel_event=None):
    pending = set(futures)
    while pending:
        check_cancelled(cancel_event)
        done, pending = concurrent.futures.wait(pending, timeout=0.1,
                                                return_when=concurrent.futures.FIRST_COMPLETED)
        for future in done:
            check_cancelled(cancel_event)
            yield future


def run_process(command, cancel_event=None, timeout=300):
    check_cancelled(cancel_event)
    started = time.monotonic()
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding='utf-8', errors='replace',
                          creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)) as process:
        try:
            while True:
                check_cancelled(cancel_event)
                if time.monotonic() - started > timeout:
                    raise TimeoutError('媒体处理超时')
                try:
                    stdout, stderr = process.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
            return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
