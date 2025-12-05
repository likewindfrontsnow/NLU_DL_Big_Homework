import os
import concurrent.futures
import re
import shutil
from typing import List, Tuple
from .frame_extractor import FrameExtractor
from ai_services.vlm_api import analyze_image

class VisualManager:
    def __init__(self):
        self.extractor = FrameExtractor()

    def _parse_timestamp_from_filename(self, filename: str, interval: int) -> str:
        try:
            match = re.search(r'frame_(\d+)', filename)
            if match:
                index = int(match.group(1))
                total_seconds = index * interval
                return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"
        except Exception:
            pass
        return "未知时间"

    def process_video_for_visual_summary(
        self, 
        video_path: str, 
        interval_seconds: int = 45, 
        mode: str = "general",
        max_workers: int = 4,
        model: str = "qwen3-vl-plus",
        keep_intermediate_files: bool = False
    ) -> str:
        base_dir = os.path.dirname(os.path.abspath(video_path))
        temp_frame_dir = os.path.join("temp", "vlm_frames_output")
        
        frames = self.extractor.extract_smart_frames(
            video_path, 
            interval_seconds=interval_seconds, 
            output_dir=temp_frame_dir
        )
        
        if not frames:
            return "【视觉分析报告】\n(未提取到有效画面)"

        results: List[Tuple[str, str]] = []
        
        def _analyze_task(frame_path):
            timestamp = self._parse_timestamp_from_filename(os.path.basename(frame_path), interval_seconds)
            try:
                desc = analyze_image(frame_path, mode=mode, model=model)
                return (timestamp, desc)
            except Exception:
                return (timestamp, "[该帧分析失败]")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_frame = {executor.submit(_analyze_task, fp): fp for fp in frames}
            
            for future in concurrent.futures.as_completed(future_to_frame):
                try:
                    results.append(future.result())
                except Exception:
                    pass

        def time_str_to_seconds(t_str):
            try:
                if t_str == "未知时间": return 999999
                m, s = map(int, t_str.split(':'))
                return m * 60 + s
            except:
                return 999999

        results.sort(key=lambda x: time_str_to_seconds(x[0]))

        summary_lines = [f"\n### 📺 视频视觉内容分析报告 (模式: {mode} | 模型: {model})"]
        summary_lines.append(f"共分析关键帧: {len(frames)} 张 | 采样间隔: {interval_seconds}秒\n")
        
        for timestamp, desc in results:
            summary_lines.append(f"**[{timestamp}]**\n{desc}\n")
            
        final_report = "\n".join(summary_lines)

        if not keep_intermediate_files:
            try:
                if os.path.exists(temp_frame_dir):
                    shutil.rmtree(temp_frame_dir)
            except Exception:
                pass
        else:
            print(f"  > [VLM] 截图已保留至: {temp_frame_dir}")

        return final_report