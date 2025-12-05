import subprocess
import os
import math
import concurrent.futures
from core.utils import retry

def _get_binary_path(binary_name: str) -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_bin = os.path.join(base_dir, "bin", f"{binary_name}.exe")
    return local_bin if os.path.exists(local_bin) else binary_name

def get_media_duration(media_path: str) -> float | None:
    ffprobe_cmd = _get_binary_path("ffprobe")
    command = [
        ffprobe_cmd, 
        '-v', 'error', 
        '-show_entries', 'format=duration', 
        '-of', 'default=noprint_wrappers=1:nokey=1', 
        media_path
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        return float(result.stdout)
    except Exception as e:
        print(f"获取媒体时长失败: {e}")
        return None

@retry(max_retries=3, delay=2, allowed_exceptions=(subprocess.CalledProcessError,))
def _process_chunk(args) -> str | None:
    media_path, output_dir, chunk_duration, i, num_chunks = args
    start_time = i * chunk_duration
    output_filename = os.path.join(output_dir, f"chunk_{i+1:03d}.mp3")
    ffmpeg_cmd = _get_binary_path("ffmpeg")
    
    command = [
        ffmpeg_cmd, '-i', media_path, 
        '-ss', str(start_time), 
        '-t', str(chunk_duration), 
        '-vn', '-acodec', 'libmp3lame', 
        '-q:a', '2', '-y', output_filename
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        return output_filename
    except subprocess.CalledProcessError as e:
        print(f"处理音频块 {i+1} 失败: {e.stderr}")
        raise e

def split_media_to_audio_chunks_generator(media_path: str, output_dir: str, chunk_duration: int = 600):
    if not os.path.exists(media_path):
        yield 'error', f"媒体文件不存在: {media_path}", None
        return
    
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        yield 'error', f"创建输出目录失败: {e}", None
        return

    duration = get_media_duration(media_path)
    if not duration:
        yield 'error', "无法获取媒体时长", None
        return

    num_chunks = math.ceil(duration / chunk_duration)
    if num_chunks == 0:
        yield 'result', []
        return
        
    tasks_args = [(media_path, output_dir, chunk_duration, i, num_chunks) for i in range(num_chunks)]
    output_files = []
    completed_count = 0
    
    max_workers = min(os.cpu_count() or 1, 8)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_args = {executor.submit(_process_chunk, args): args for args in tasks_args}
        
        for future in concurrent.futures.as_completed(future_to_args):
            try:
                result = future.result()
                if result:
                    output_files.append(result)
            except Exception as e:
                yield 'error', f"音频块处理失败: {e}", None
                executor.shutdown(wait=False, cancel_futures=True)
                return

            completed_count += 1
            yield 'progress', completed_count, num_chunks

    if len(output_files) != num_chunks:
        yield 'error', "未能生成所有音频块", None
        return
    
    yield 'result', sorted(output_files)