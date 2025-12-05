# vlm_api.py
import os
import dashscope
from http import HTTPStatus
from typing import Literal, Optional
from config import DASHSCOPE_API_KEY
from utils import retry

# 初始化 API Key
if DASHSCOPE_API_KEY:
    dashscope.api_key = DASHSCOPE_API_KEY
else:
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

class VLMError(Exception):
    """VLM 模块自定义异常"""
    pass

# --- 预置的高级 Prompts ---
VLM_PROMPTS = {
    "general": (
        "请详细描述这张图片的内容，包括主要物体、文字信息（如果有）以及场景氛围。"
    ),
    
    "ppt": (
        "这张图片是课程的PPT幻灯片。\n"
        "任务：请充当专业的OCR和内容整理员，将图片内容转换为结构化的文本。\n"
        "要求：\n"
        "1. **精准OCR**：按原义提取所有可见文字，包括标题、正文和备注。\n"
        "2. **保留结构**：使用Markdown格式（# 标题, - 列表）还原PPT的层级结构。\n"
        "3. **图表描述**：如果遇到图表（如柱状图、流程图），请用文字详细概括其核心数据和趋势。\n"
        "4. **公式处理**：如果包含数学公式，必须转换为标准的LaTeX格式（例如 $E=mc^2$）。\n"
        "5. **排除干扰**：忽略软件界面（如PowerPoint工具栏）或无关背景，只关注内容本身。"
    ),
    
    "handwriting": (
        "这张图片是老师在黑板/白板上的手写板书。\n"
        "任务：请充当学术助教，识别并整理老师的板书内容。\n"
        "要求：\n"
        "1. **手写识别**：尽可能准确地辨认手写字体。对于模糊不清的部分，根据上下文进行合理推断，如果无法推断则标记为 [无法识别]。\n"
        "2. **逻辑还原**：板书通常是按照推导逻辑书写的（从左到右，从上到下）。请还原这个逻辑流，而不是杂乱地堆砌文字。\n"
        "3. **数学公式**：这是核心要求。所有数学符号、方程、推导过程必须严格转换为 LaTeX 格式。\n"
        "4. **绘图说明**：如果老师画了示意图（如几何图形、函数图像），请用文字简要描述图的形状和关键标注。\n"
        "5. **去噪**：忽略黑板擦痕、反光或讲师的身体遮挡，只输出板书内容。"
    )
}

@retry(max_retries=3, delay=2, backoff_factor=2)
def analyze_image(
    image_path: str, 
    mode: Literal["general", "ppt", "handwriting"] = "general",
    custom_prompt: Optional[str] = None,
    model: str = "qwen3-vl-plus"
) -> str:
    """
    调用视觉大模型 (VLM) 对图片进行分析。

    :param image_path: 本地图片的绝对路径
    :param mode: 模式选择 ("general" | "ppt" | "handwriting")，决定使用哪种预置 Prompt
    :param custom_prompt: 如果提供，将覆盖 mode 对应的默认 Prompt
    :param model: 使用的模型名称
    :return: 模型的文本响应
    """
    
    # 1. 校验文件
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"VLM Error: 找不到图片文件 -> {image_path}")

    # 2. 确定 Prompt
    # 优先级: custom_prompt > VLM_PROMPTS[mode] > VLM_PROMPTS["general"]
    if custom_prompt:
        final_prompt = custom_prompt
    else:
        final_prompt = VLM_PROMPTS.get(mode, VLM_PROMPTS["general"])

    # 3. 构造本地文件协议路径
    abs_path = os.path.abspath(image_path)
    image_url = f"file://{abs_path}"

    print(f"  > [VLM] 正在分析图片: {os.path.basename(image_path)}")
    print(f"  > 模式: {mode} | 模型: {model}")

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": image_url},
                    {"text": final_prompt},
                ]
            }
        ]

        response = dashscope.MultiModalConversation.call(
            model=model,
            messages=messages,
            result_format="message"
        )

        if response.status_code == HTTPStatus.OK:
            if response.output and response.output.choices:
                content_list = response.output.choices[0].message.content
                full_text = ""
                # 兼容处理：有些SDK版本 content 是列表，有些直接是字符串
                if isinstance(content_list, list):
                    for item in content_list:
                        if isinstance(item, dict) and "text" in item:
                            full_text += item["text"]
                elif isinstance(content_list, str):
                    full_text = content_list
                
                if not full_text:
                    # 再次尝试兜底
                    try:
                        full_text = response.output.choices[0].message.content
                    except:
                        pass

                if not full_text:
                     raise VLMError("API 返回成功但内容为空")

                print(f"  > [VLM] 分析成功 (长度: {len(full_text)} 字符)")
                return full_text
            else:
                raise VLMError("API 响应结构异常，未找到 choices")
        else:
            error_msg = f"Code: {response.code}, Message: {response.message}"
            print(f"  > ❌ [VLM] 调用失败: {error_msg}")
            
            if response.code == "Throttling.RateQuota" or response.status_code == 429:
                raise Exception(f"VLM Rate Limit: {error_msg}")
            
            raise VLMError(f"DashScope API Error: {error_msg}")

    except Exception as e:
        print(f"  > ⚠️ [VLM] 异常: {e}")
        raise e

# --- 单元测试 ---
if __name__ == "__main__":
    print("--- 开始测试 vlm_api.py (多模式) ---")
    
    # 修改这里的路径为你的一张真实图片路径
    test_img = r"D:\Documents\111study\coding\Python\NLU-DL\NLU_DL_Big_Homework\test_ppt.jpg" 
    
    if not os.path.exists(test_img):
        print(f"提示: 请在当前目录下放置一张名为 '{test_img}' 的图片用于测试 PPT 模式。")
    else:
        try:
            # 测试 PPT 模式
            print("\n🔍 测试 PPT 模式:")
            res_ppt = analyze_image(test_img, mode="ppt")
            print("-" * 20)
            print(res_ppt[:200] + "..." if len(res_ppt) > 200 else res_ppt)

            # 测试 板书 模式 (用同一张图演示调用，实际场景请换图)
            # print("\n🔍 测试 板书 模式:")
            # res_board = analyze_image(test_img, mode="handwriting")
            # print("-" * 20)
            # print(res_board[:200] + "...")
            
        except Exception as err:
            print(f"\n❌ 测试失败: {err}")