import os
import concurrent.futures
import re
import shutil
from typing import List, Tuple
from .frame_extractor import FrameExtractor
from ai_services.vlm_api import analyze_image
from PIL import Image

class VisualManager:
    def __init__(self):
        self.extractor = FrameExtractor()

    def _parse_timestamp_from_filename(self, filename: str, interval: int) -> str:
        try:
            match = re.search(r'frame_(\d+)', filename)
            if match:
                index = int(match.group(1))
                total_seconds = (index - 1) * interval
                if total_seconds < 0: total_seconds = 0
                return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"
        except Exception:
            pass
        return "未知时间"

    def _calculate_image_difference(self, img_path1: str, img_path2: str) -> float:
        try:
            img1 = Image.open(img_path1).convert('L').resize((64, 64))
            img2 = Image.open(img_path2).convert('L').resize((64, 64))
            
            p1 = list(img1.getdata())
            p2 = list(img2.getdata())
            
            diff_sum = sum(abs(a - b) for a, b in zip(p1, p2))
            mean_diff = diff_sum / len(p1)
            return mean_diff
        except Exception:
            return 999.0

    def process_video_generator(
        self, 
        video_path: str, 
        interval_seconds: int = 60, 
        mode: str = "general",
        max_workers: int = 10,
        model: str = "qwen3-vl-plus",
        backup_model: str = None,
        keep_intermediate_files: bool = False,
        insert_images: bool = False,
        image_output_dir: str = None
    ):
        base_dir = os.path.dirname(os.path.abspath(video_path))
        temp_frame_dir = os.path.join("temp", "vlm_frames_output")
        
        extraction_gen = self.extractor.extract_frames_generator(
            video_path, 
            interval_seconds=interval_seconds, 
            output_dir=temp_frame_dir
        )
        
        frames = []
        
        for event, data in extraction_gen:
            if event == 'progress':
                # data is (current, total)
                yield "stage_progress", "extracting", data[0], data[1]
            elif event == 'result':
                frames = data
            elif event == 'error':
                yield "log", 0, f"抽帧错误: {data}"
                
        if not frames:
            yield "result", "【视觉分析报告】\n(未提取到有效画面)", None
            return

        yield "log", 0, "正在进行图像去重，剔除静止画面..."
        
        unique_frames = []
        if frames:
            unique_frames.append(frames[0])
            last_kept_frame = frames[0]
            dropped_count = 0
            SIMILARITY_THRESHOLD = 12.0 
            
            for i in range(1, len(frames)):
                current_frame = frames[i]
                diff = self._calculate_image_difference(last_kept_frame, current_frame)
                
                if diff > SIMILARITY_THRESHOLD:
                    unique_frames.append(current_frame)
                    last_kept_frame = current_frame
                else:
                    dropped_count += 1
            
            if dropped_count > 0:
                yield "log", 0, f"✅ 去重完成：原 {len(frames)} 帧 -> 现 {len(unique_frames)} 帧 (过滤了 {dropped_count} 张重复画面)"
            frames = unique_frames

        total_frames = len(frames)
        yield "stage_progress", "analyzing", 0, total_frames

        results: List[Tuple[str, str, str]] = []
        
        def _analyze_task(frame_path):
            timestamp = self._parse_timestamp_from_filename(os.path.basename(frame_path), interval_seconds)
            
            models_to_try = [model]
            if backup_model and backup_model != model and backup_model != "(无备选)":
                models_to_try.append(backup_model)
            
            last_error = None
            for m in models_to_try:
                try:
                    desc = analyze_image(frame_path, mode=mode, model=m)
                    return (timestamp, desc, frame_path)
                except Exception as e:
                    last_error = e
                    continue
            return (timestamp, f"[该帧分析失败: {str(last_error)}]", frame_path)

        completed_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_frame = {executor.submit(_analyze_task, fp): fp for fp in frames}
            
            for future in concurrent.futures.as_completed(future_to_frame):
                completed_count += 1
                try:
                    res = future.result()
                    results.append(res)
                except Exception:
                    pass
                
                yield "stage_progress", "analyzing", completed_count, total_frames

        def time_str_to_seconds(t_str):
            try:
                if t_str == "未知时间": return 999999
                m, s = map(int, t_str.split(':'))
                return m * 60 + s
            except:
                return 999999

        results.sort(key=lambda x: time_str_to_seconds(x[0]))

        summary_lines = [f"\n### 📺 视频视觉内容分析报告 (模式: {mode} | 主模型: {model})"]
        summary_lines.append(f"共分析关键帧: {len(frames)} 张 | 采样间隔: {interval_seconds}秒\n")
        
        if insert_images and image_output_dir:
            try:
                os.makedirs(image_output_dir, exist_ok=True)
            except Exception:
                pass

        for timestamp, desc, frame_src in results:
            item_text = f"**[{timestamp}]**\n{desc}\n"
            if insert_images and image_output_dir and os.path.exists(frame_src):
                try:
                    file_name = os.path.basename(frame_src)
                    safe_name = f"{timestamp.replace(':', '')}_{file_name}"
                    dst_path = os.path.join(image_output_dir, safe_name)
                    shutil.copy(frame_src, dst_path)
                    rel_dir_name = os.path.basename(image_output_dir)
                    rel_path = f"{rel_dir_name}/{safe_name}"
                    item_text += f"\n![关键帧截图]({rel_path})\n"
                except Exception:
                    pass
            summary_lines.append(item_text)
            
        final_report = "\n".join(summary_lines)

        if not keep_intermediate_files:
            try:
                if os.path.exists(temp_frame_dir):
                    shutil.rmtree(temp_frame_dir)
            except Exception:
                pass

        yield "result", final_report, None