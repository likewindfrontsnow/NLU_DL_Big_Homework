# test_integration.py
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from visual_manager import VisualManager
from frame_extractor import FrameExtractor
from vlm_api import analyze_image
import os
import sys

def test_full_pipeline():
    print("🚀 开始集成测试...")
    
    # 1. 检查测试视频
    # 请确保这里有一个真实的短视频文件，或修改路径
    test_video = "test_video.mp4" 
    
    if not os.path.exists(test_video):
        print(f"❌ 错误：找不到测试视频 '{test_video}'。")
        print("💡 建议：请在当前目录下放一个小的 mp4 文件，重命名为 test_video.mp4")
        return

    print(f"✅ 找到测试视频: {test_video}")

    # 2. 测试 FrameExtractor (抽帧)
    print("\n[1/3] 测试抽帧模块...")
    extractor = FrameExtractor()
    temp_dir = "test_temp_frames"
    try:
        frames = extractor.extract_smart_frames(test_video, interval_seconds=10, output_dir=temp_dir)
        if frames and len(frames) > 0:
            print(f"✅ 抽帧成功！生成了 {len(frames)} 张图片。")
            print(f"   第一张图路径: {frames[0]}")
        else:
            print("❌ 抽帧失败：未生成图片。请检查 FFmpeg 配置。")
            return
    except Exception as e:
        print(f"❌ 抽帧模块抛出异常: {e}")
        return

    # 3. 测试 VLM API (单张图片分析)
    print("\n[2/3] 测试 VLM API 连接...")
    if frames:
        test_img = frames[0]
        try:
            print(f"   正在分析图片: {test_img} (模式: general)...")
            result = analyze_image(test_img, mode="general")
            print(f"✅ VLM 调用成功！返回长度: {len(result)}")
            print(f"   内容预览: {result[:50]}...")
        except Exception as e:
            print(f"❌ VLM 调用失败: {e}")
            print("💡 建议：检查 .env 文件中的 DASHSCOPE_API_KEY 是否有效。")
            return

    # 4. 测试 VisualManager (完整流程)
    print("\n[3/3] 测试 VisualManager (并发分析与汇总)...")
    manager = VisualManager()
    try:
        report = manager.process_video_for_visual_summary(
            test_video, 
            interval_seconds=30, # 间隔设大点，测试速度快
            mode="ppt",
            max_workers=2
        )
        
        print("✅ VisualManager 运行完成！报告预览：")
        print("-" * 30)
        print(report[:300] + "..." if len(report) > 300 else report)
        print("-" * 30)
        
        # 检查是否包含时间戳
        if "**[00:00]**" in report or "**[00:30]**" in report:
             print("✅ 报告中包含正确的时间戳格式。")
        else:
             print("⚠️ 警告：报告中似乎未检测到预期的 [MM:SS] 时间戳，请检查正则匹配逻辑。")
             
    except Exception as e:
        print(f"❌ VisualManager 运行失败: {e}")

    # 清理测试产生的临时文件
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"\n🧹 已清理测试临时目录: {temp_dir}")

if __name__ == "__main__":
    test_full_pipeline()