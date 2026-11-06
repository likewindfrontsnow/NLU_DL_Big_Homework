import subprocess
import os
import concurrent.futures
from core.tasks import TaskCancelled, completed_futures, run_process

class FrameExtractor:
    def _get_binary_path(self, binary_name: str) -> str:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_script_dir)
        local_bin = os.path.join(project_root, "bin", binary_name + ".exe")
        
        if os.path.exists(local_bin):
            return local_bin
        
        return binary_name

    def get_video_duration(self, video_path: str, cancel_event=None) -> float:
        if not os.path.exists(video_path):
            return 0.0

        ffprobe_cmd = self._get_binary_path("ffprobe")
        command = [
            ffprobe_cmd,
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        
        try:
            result = run_process(command, cancel_event=cancel_event)
            return float(result.stdout.strip())
        except TaskCancelled:
            raise
        except Exception:
            return 0.0

    def _extract_single_frame_task(self, args, cancel_event=None):
        ffmpeg_cmd, video_path, time_sec, output_path = args
        command = [
            ffmpeg_cmd,
            '-ss', str(time_sec),
            '-i', video_path,
            '-frames:v', '1',
            '-q:v', '2',
            '-y',
            output_path
        ]
        
        try:
            run_process(command, cancel_event=cancel_event)
            return output_path
        except TaskCancelled:
            raise
        except Exception:
            return None

    def extract_frames_generator(self, video_path: str, interval_seconds: int = 60, output_dir: str = "extracted_frames", max_workers: int = 12, cancel_event=None):
        if not os.path.exists(video_path):
            yield 'error', f"找不到视频文件: {video_path}"
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        duration = self.get_video_duration(video_path, cancel_event)
        if duration <= 0:
            yield 'error', "无法获取视频时长"
            return

        timestamps = []
        t = 0.0
        while t < duration - 0.5:
            timestamps.append(t)
            t += interval_seconds
        
        if not timestamps:
            timestamps = [0.0]

        total_frames = len(timestamps)
        ffmpeg_cmd = self._get_binary_path("ffmpeg")
        
        tasks = []
        for i, t_sec in enumerate(timestamps):
            out_name = os.path.join(output_dir, f"frame_{i+1:03d}.jpg")
            tasks.append((ffmpeg_cmd, video_path, t_sec, out_name))

        extracted_files = []
        completed_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self._extract_single_frame_task, task_args, cancel_event): task_args[3] for task_args in tasks}
            
            for future in completed_futures(future_to_file, cancel_event):
                completed_count += 1
                out_path = future.result()
                
                if out_path and os.path.exists(out_path):
                    extracted_files.append(out_path)
                
                yield 'progress', (completed_count, total_frames)

        extracted_files.sort()
        yield 'result', extracted_files