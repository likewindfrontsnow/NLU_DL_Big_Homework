import subprocess
import os

class FrameExtractor:
    def _get_binary_path(self, binary_name: str) -> str:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_script_dir)
        local_bin = os.path.join(project_root, "bin", binary_name + ".exe")
        
        if os.path.exists(local_bin):
            return local_bin
        
        return binary_name

    def extract_smart_frames(self, video_path: str, interval_seconds: int = 30, scene_threshold: float = 0.2, output_dir: str = "extracted_frames"):
        if not os.path.exists(video_path):
            print(f"❌ 错误：找不到视频文件: {video_path}")
            return []

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        ffmpeg_cmd = self._get_binary_path("ffmpeg")
        output_pattern = os.path.join(output_dir, "frame_%03d.jpg")

        command = [
            ffmpeg_cmd,
            '-i', video_path,
            '-vf', f'fps=1/{interval_seconds}',
            '-q:v', '2',
            '-y',
            output_pattern
        ]

        print(f"  > [FrameExtractor] 正在抽帧 (每 {interval_seconds} 秒一帧)...")

        try:
            subprocess.run(
                command, 
                check=True, 
                capture_output=True, 
                text=True, 
                encoding='utf-8', 
                errors='ignore'
            )
            print("  > ✅ 抽帧完成！")
            
            frames = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.jpg')]
            return frames

        except FileNotFoundError:
            print("\n❌ 严重错误：系统找不到 'ffmpeg' 命令。")
            print(f"  试图查找的路径包括全局 PATH 和: {os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'bin')}")
            return []
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg 执行出错: {e.stderr}")
            return []
        except Exception as e:
            print(f"❌ 发生未知错误: {e}")
            return []