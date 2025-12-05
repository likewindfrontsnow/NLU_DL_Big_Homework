# video_processor/visual_manager.py
import os
import concurrent.futures
import re
import shutil
from typing import List, Tuple
from .frame_extractor import FrameExtractor
from ..ai_services.vlm_api import analyze_image

class VisualManager:
    def __init__(self):
        self.extractor = FrameExtractor()

    def _parse_timestamp_from_filename(self, filename: str, interval: int) -> str:
        """
        根据文件名反推大概的时间戳。
        文件名通常由 FrameExtractor 生成，格式如 frame_001.jpg
        """
        try:
            # 提取数字部分 frame_001.jpg -> 1
            match = re.search(r'frame_(\d+)', filename)
            if match:
                index = int(match.group(1))
                # 假设 frame_001 是第1个间隔点 (例如30s), frame_000 是0s(如果有的话)
                # FrameExtractor 通常从 1 开始计数
                total_seconds = index * interval
                
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                return f"{minutes:02d}:{seconds:02d}"
        except Exception:
            pass
        return "未知时间"

    def process_video_for_visual_summary(
        self, 
        video_path: str, 
        interval_seconds: int = 45, 
        mode: str = "general",
        max_workers: int = 4
    ) -> str:
        """
        处理视频：抽帧 -> VLM分析 -> 汇总生成视觉报告
        
        :param video_path: 视频路径
        :param interval_seconds: 抽帧间隔 (建议设大一点，如 45s 或 60s，节省 Token)
        :param mode: VLM 模式 ("ppt" | "handwriting" | "general")
        :param max_workers: 并发请求数量 (Qwen-VL 限流较严格，建议 3-5)
        :return: 格式化的视觉内容描述文本
        """
        
        print(f"🎬 [VisualManager] 开始处理视频视觉信息: {os.path.basename(video_path)}")
        
        # 1. 准备临时文件夹
        # 在视频同级目录下创建 temp_frames_for_analysis
        base_dir = os.path.dirname(os.path.abspath(video_path))
        temp_frame_dir = os.path.join(base_dir, "temp_frames_for_analysis")
        
        # 2. 调用抽帧模块
        frames = self.extractor.extract_smart_frames(
            video_path, 
            interval_seconds=interval_seconds, 
            output_dir=temp_frame_dir
        )
        
        if not frames:
            return "【视觉分析报告】\n(未提取到有效画面，可能视频全黑或文件损坏)"

        print(f"  > 准备分析 {len(frames)} 张关键帧 (模式: {mode}, 并发: {max_workers})...")
        
        # 3. 并发调用 VLM 进行分析
        results: List[Tuple[str, str]] = [] # [(timestamp, description), ...]
        
        def _analyze_task(frame_path):
            # 计算该帧对应的时间戳
            timestamp = self._parse_timestamp_from_filename(os.path.basename(frame_path), interval_seconds)
            try:
                # 调用 VLM API
                desc = analyze_image(frame_path, mode=mode)
                return (timestamp, desc)
            except Exception as e:
                print(f"  > ⚠️ 帧 {os.path.basename(frame_path)} 分析失败: {e}")
                return (timestamp, "[该帧分析失败]")

        # 使用线程池并发处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_frame = {executor.submit(_analyze_task, fp): fp for fp in frames}
            
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_frame):
                try:
                    res = future.result()
                    results.append(res)
                    completed_count += 1
                    # 简单的进度打印
                    # print(f"  > [VLM 进度] {completed_count}/{len(frames)} 帧处理完毕")
                except Exception as e:
                    print(f"  > ❌ 线程异常: {e}")

        # 4. 整理结果 (按时间顺序排序)
        def time_str_to_seconds(t_str):
            try:
                if t_str == "未知时间": return 999999
                m, s = map(int, t_str.split(':'))
                return m * 60 + s
            except:
                return 999999

        results.sort(key=lambda x: time_str_to_seconds(x[0]))

        # 5. 生成最终文本报告
        summary_lines = [f"\n### 📺 视频视觉内容分析报告 (模式: {mode})"]
        summary_lines.append(f"共分析关键帧: {len(frames)} 张 | 采样间隔: {interval_seconds}秒\n")
        
        for timestamp, desc in results:
            item = f"**[{timestamp}]**\n{desc}\n"
            summary_lines.append(item)
            
        final_report = "\n".join(summary_lines)
        
        # 6. 清理临时图片
        try:
            if os.path.exists(temp_frame_dir):
                shutil.rmtree(temp_frame_dir)
                print(f"  > 已清理临时帧目录: {temp_frame_dir}")
        except Exception as e:
            print(f"  > ⚠️ 临时目录清理失败: {e}")

        return final_report

# --- 测试代码 ---
if __name__ == "__main__":
    # 简单的本地测试逻辑
    print("--- VisualManager 测试模式 ---")
    
    # 请修改为你的真实测试视频路径
    test_video_path = r"test_video.mp4" 
    
    if os.path.exists(test_video_path):
        manager = VisualManager()
        # 测试 PPT 模式
        report = manager.process_video_for_visual_summary(
            test_video_path, 
            interval_seconds=30, 
            mode="ppt",
            max_workers=2
        )
        print("\n" + "="*40)
        print(report)
        print("="*40)
    else:
        print(f"提示: 未找到测试视频 '{test_video_path}'，跳过测试。")