import os
import dashscope
from http import HTTPStatus
from typing import Literal, Optional
from core.config import DASHSCOPE_API_KEY
from core.utils import retry

if DASHSCOPE_API_KEY:
    dashscope.api_key = DASHSCOPE_API_KEY
else:
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

class VLMError(Exception):
    pass

VLM_PROMPTS = {
    "general": (
        "请详细描述这张图片的内容，包括主要物体、文字信息（如果有）以及场景氛围。"
    ),
    "lecture_mixed": (
        "这张图片来自于课程教学视频的截屏。画面内容可能是 **PowerPoint 幻灯片**、**老师的板书 (黑板/白板)**，或者是 **二者的混合**。\n"
        "任务：请充当专业的学术 OCR 助手，精准提取并整理画面中的核心知识信息。\n"
        "**核心要求**：\n"
        "1. **内容识别与区分**：\n"
        "   - 如果是 **PPT**：按原标题层级还原文字，描述关键图表含义。\n"
        "   - 如果是 **板书**：请仔细辨认手写体，按推导逻辑（从左到右、从上到下）整理内容。\n"
        "   - 如果是 **混合场景**：请同时提取 PPT 和板书的内容，并简要注明来源（如“[PPT内容]... [板书补充]...”）。\n"
        "2. **数学公式标准化**：\n"
        "   - 画面中出现的所有数学公式、变量符号，**必须** 转换为标准的 **LaTeX** 格式（例如 $E=mc^2$）。\n"
        "   - 即使是手写模糊的公式，也要结合上下文逻辑进行推断和补全。\n"
        "3. **去除无关干扰**：\n"
        "   - 忽略视频播放器的界面 UI、无关的人物背景或遮挡物，只关注教学内容本身。\n"
        "4. **输出格式**：\n"
        "   - 直接输出整理后的 Markdown 文本，不要包含“这是一张PPT”等废话，直接罗列知识点。"
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
    mode: Literal["general", "ppt", "handwriting", "lecture_mixed"] = "general",
    custom_prompt: Optional[str] = None,
    model: str = "qwen3-vl-plus"
) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"VLM Error: 找不到图片文件 -> {image_path}")

    final_prompt = custom_prompt if custom_prompt else VLM_PROMPTS.get(mode, VLM_PROMPTS["general"])
    image_url = f"file://{os.path.abspath(image_path)}"

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
                
                if isinstance(content_list, list):
                    for item in content_list:
                        if isinstance(item, dict) and "text" in item:
                            full_text += item["text"]
                elif isinstance(content_list, str):
                    full_text = content_list
                
                if not full_text:
                    try:
                        full_text = response.output.choices[0].message.content
                    except Exception:
                        pass

                if not full_text:
                     raise VLMError("API 返回成功但内容为空")

                return full_text
            else:
                raise VLMError("API 响应结构异常，未找到 choices")
        else:
            error_msg = f"Code: {response.code}, Message: {response.message}"
            
            if response.code == "Throttling.RateQuota" or response.status_code == 429:
                raise Exception(f"VLM Rate Limit: {error_msg}")
            
            raise VLMError(f"DashScope API Error: {error_msg}")

    except Exception as e:
        raise e