# app.py
import streamlit as st
import os
import collections 
from main import main_process_generator
from config import LLM_CONFIG 

st.set_page_config(page_title="智能笔记 Agent", layout="wide")
st.title("👨‍💻 智能内容生成 Agent")
st.markdown("上传您的视频、音频或文本文档，即可自动生成结构化笔记、Q&A 或测验。")

# --- (重大修改) ---
# 1. 升级 PLATFORMS 数据结构，包含 "url" 和 "model"
PLATFORMS = collections.OrderedDict([
    ("DashScope (通义Qwen)", {"url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus"}),
    ("Baidu Qianfan (文心ERNIE)", {"url": "https://qianfan.baidubce.com/v1", "model": "ernie-bot-4.0"}),
    ("Volcano Engine (豆包Doubao)", {"url": "https://ark.cn-beijing.volces.com/api/v3", "model": "Doubao-pro-32k"}),
    ("Zhipu AI (智谱GLM)", {"url": "https://open.bigmodel.cn/api/paas/v4", "model": "glm-4"}),
    ("Moonshot (Kimi)", {"url": "https://api.moonshot.cn/v1", "model": "moonshot-v1-8k"}),
    ("OpenAI (GPT-4/3.5)", {"url": "https://api.openai.com/v1", "model": "gpt-4o"}),
    ("Groq (Llama 3/Mixtral)", {"url": "https://api.groq.com/openai/v1", "model": "llama3-8b-8192"}),
    ("Google (Gemini)", {"url": "", "model": "gemini-1.5-pro-latest"}), # 提示用户使用代理
    ("Anthropic (Claude)", {"url": "", "model": "claude-3-sonnet-20240229"}), # 提示用户使用代理
    ("Custom (自定义)", {"url": "", "model": ""}) # 自定义
])
platform_options = list(PLATFORMS.keys())


# --- (重大修改) 状态管理 ---

# 2. 升级回调函数 (现在也更新 model)
def platform_changed():
    """当用户更改平台选择时，此函数被调用"""
    selected_platform = st.session_state.platform_selector
    platform_data = PLATFORMS[selected_platform]
    
    # (1) 更新 Base URL
    new_base_url = platform_data["url"]
    if not new_base_url: # 对于 Custom、Claude 等
        # 保持用户在 .env 中设置的 URL，以支持自定义代理
        new_base_url = LLM_CONFIG.get('base_url', '') 
    st.session_state.user_base_url_input = new_base_url
    
    # (2) 更新 Model
    new_model = platform_data["model"]
    if not new_model: # 对于 Custom
         new_model = LLM_CONFIG.get('model', '') # 保持 .env 的设置
    st.session_state.user_model_input = new_model
    
    # (3) 更新服务商名称
    if selected_platform == "Custom (自定义)":
        st.session_state.user_provider_name_input = LLM_CONFIG.get('provider_name', 'LLM')
    else:
        st.session_state.user_provider_name_input = selected_platform

# 3. (重大修复) 首次加载时初始化所有 session_state 键
def get_default_platform_name(env_url):
    """(已修复) 辅助函数：根据 .env 的 URL 查找平台名称"""
    if not env_url:
        return "Custom (自定义)"
    for name, data in PLATFORMS.items():
        if data["url"] == env_url:
            return name
    # 如果 .env 的 URL 不在预设中，说明它就是 "Custom"
    return "Custom (自定义)"

# (已修复) 确保只在首次加载时执行
if 'platform_selector' not in st.session_state:
    # (1) 平台选择器：基于 .env 的 URL 动态设置
    env_url = LLM_CONFIG.get('base_url')
    st.session_state.platform_selector = get_default_platform_name(env_url)

    # (2) 其他输入框：严格使用 .env 的值进行初始化
    st.session_state.user_base_url_input = LLM_CONFIG.get('base_url', '')
    st.session_state.user_api_key_input = LLM_CONFIG.get('api_key', '')
    st.session_state.user_model_input = LLM_CONFIG.get('model', '')
    
    # (3) 服务商名称：基于 .env 或平台名称
    if st.session_state.platform_selector == "Custom (自定义)":
        st.session_state.user_provider_name_input = LLM_CONFIG.get('provider_name', 'LLM')
    else:
        # 如果 URL 匹配，则使用平台名称
        st.session_state.user_provider_name_input = st.session_state.platform_selector
# --- 结束修改 ---


default_provider_name = LLM_CONFIG.get('provider_name', 'LLM')
st.info(f"💡 **提示**: 视频/音频文件将使用本地 Whisper 转录。笔记生成默认调用 **{default_provider_name}** API（可在侧边栏修改）。")

with st.sidebar:
    st.header("⚙️ 核心配置")
    st.subheader("LLM API 配置")
    st.markdown("您可以覆盖 `.env` 中的默认配置。")

    # --- (修改) 控件现在绑定到 session_state ---

    # 1. 平台选择
    st.selectbox(
        "选择 API 平台:",
        options=platform_options,
        key="platform_selector", # 绑定到 state
        on_change=platform_changed, # 注册回调
        help="选择一个预设平台将自动填充 Base URL 和推荐 Model。"
    )

    # 2. Base URL
    user_base_url = st.text_input(
        "API Base URL:",
        key="user_base_url_input" # 只使用 key
    )
    
    # (新增) 将警告逻辑移到控件外部
    if PLATFORMS.get(st.session_state.platform_selector, {}).get("url") == "" and st.session_state.platform_selector != "Custom (自定义)":
         st.warning(f"“{st.session_state.platform_selector}” 的原生 API 与当前代码不兼容。\n\n请确保上方是一个**代理 Base URL**。")

    # 3. API Key
    user_api_key = st.text_input(
        "API Key:",
        type="password",
        key="user_api_key_input",
        help="您输入的 API Key 将优先于 .env 文件中的设置。"
    )

    # 4. Model (修改)
    user_model = st.text_input(
        "Model:",
        key="user_model_input", # (修改) 绑定到 state
        help="要使用的模型名称, e.g., 'qwen-plus', 'glm-4', 'gpt-4o'."
    )
    
    # 5. Provider Name (用于显示)
    user_provider_name = st.text_input(
        "服务商名称 (用于显示):",
        key="user_provider_name_input",
        help="用于在 UI 上显示的服务商名称。"
    )
    # --- 结束修改 ---

    st.markdown("---")
    st.header("⚙️ 生成配置")

    # ... (这部分配置保持不变) ...
    output_filename = st.text_input("请输入希望的笔记文件名 (无需后缀)", value="我的学习笔记")
    query_option = st.selectbox(
        "请选择生成内容类型:",
        ("Notes", "Q&A", "Quiz"),
        index=0,
        help="选择 'Notes' 生成结构化笔记, 'Q&A' 生成问答对, 'Quiz' 生成测验题。(目前仅 'Notes' 功能已接入)"
    )
    whisper_model_size = st.selectbox(
        "请选择语音转录模型:",
        ("tiny", "base", "small", "medium", "large"),
        index=0,
        help="模型越大，转录越准确，但速度越慢。'tiny' 最快，'large' 最准。首次使用非 'tiny' 模型时，程序会先下载模型文件（可能需要几分钟）。"
    )
    stream_output = st.toggle(
        "启用流式输出", 
        value=True, 
        help="启用后，笔记内容将实时逐字显示。禁用则会在所有内容生成后一次性显示。"
    )
    st.markdown("---")
    keep_temp_files = st.checkbox(
        "保留中间文件", 
        value=False, 
        help="勾选后将保留上传的临时文件和语音转文字生成的 `source_transcript.txt`。"
    )
    st.info("请在上方配置好参数后，上传文件开始处理。")

# ... (文件格式定义保持不变) ...
video_exts = {'mp4', 'mov','mpeg','webm'}
audio_exts = {'mp3','m4a','wav','amr','mpga'}
doc_exts = {'txt','md','mdx','markdown','pdf','html','xlsx','xls','doc','docx','csv','eml','msg','pptx','ppt','xml','epub'}
all_exts = list(video_exts | audio_exts | doc_exts)

with st.expander("查看所有支持的文件格式"):
    st.markdown(f"""
    - **视频文件**: `{', '.join(sorted(list(video_exts)))}`
    - **音频文件**: `{', '.join(sorted(list(audio_exts)))}`
    - **文档文件**: `{', '.join(sorted(list(doc_exts)))}`
    """)

uploaded_file = st.file_uploader(
    "上传视频、音频或文档", 
    type=all_exts
)

if uploaded_file is not None:
    if st.button("开始生成", use_container_width=True, type="primary"):
        
        # (修改) 现在我们从 st.session_state 读取值
        api_key_val = st.session_state.user_api_key_input
        base_url_val = st.session_state.user_base_url_input
        model_val = st.session_state.user_model_input
        provider_name_val = st.session_state.user_provider_name_input

        # (修改) 检查来自 state 的值
        if not api_key_val:
            st.error("请输入 API Key！")
        elif not base_url_val:
            st.error("请输入 API Base URL！")
        elif not model_val:
            st.error("请输入 Model！")
        else:
            st.markdown("---")
            st.subheader("处理进度")
            
            main_progress_bar = st.progress(0)
            main_progress_text = st.empty()
            sub_progress_bar = st.progress(0)
            sub_progress_text = st.empty()

            st.markdown("---")

            stream_status = "流式" if stream_output else "非流式"
            # (修改) 使用来自 state 的服务商名称
            processing_headers = {
                "Notes": f"正在生成笔记 ({provider_name_val} {stream_status})...",
                "Q&A": "正在进行 Q&A...",
                "Quiz": "正在生成测验..."
            }
            st.subheader(processing_headers.get(query_option, "正在处理..."))
            st.info(f"当前生成模式: **{query_option}** (转录模型: **{whisper_model_size}**)")
            
            classification_display = st.empty()
            llm_output_container = st.empty()
            full_llm_response = ""
            
            final_result_path = None
            processing_has_failed = False

            temp_dir = "temp_uploads"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # (修改) 从 state 组装运行时配置
            runtime_config_override = {
                "api_key": api_key_val,
                "base_url": base_url_val,
                "model": model_val,
                "provider_name": provider_name_val
            }
            # --- 结束修改 ---

            generator = main_process_generator(
                temp_file_path, 
                runtime_config_override, 
                output_filename, 
                query_option, 
                whisper_model_size,
                stream_output
            )
            
            # ... (生成器循环保持不变) ...
            for event_type, value, *rest in generator:
                text = rest[0] if rest else ""

                if event_type == "progress":
                    main_progress_bar.progress(float(value))
                    main_progress_text.info(text)
                elif event_type == "sub_progress":
                    sub_progress_bar.progress(float(value))
                    sub_progress_text.text(text)
                
                elif event_type == "llm_chunk":
                    full_llm_response += value
                    llm_output_container.markdown(full_llm_response)
                
                elif event_type == "persistent_error":
                    st.error(f"处理失败: {text}")
                    main_progress_text.error("一个关键步骤在多次重试后仍然失败，已停止处理。")
                    llm_output_container.error(f"**错误详情:**\n\n{text}")
                    if st.button("🔄 重新开始"):
                        st.experimental_rerun()
                    processing_has_failed = True
                    break
                
                elif event_type == "error":
                    st.error(text)
                    llm_output_container.error(text)
                    processing_has_failed = True
                    break

                elif event_type == "done":
                    main_progress_bar.progress(1.0)
                    sub_progress_bar.empty()
                    sub_progress_text.empty()
                    llm_output_container.markdown(full_llm_response)
                    st.success(text)
                    final_result_path = value
            
            # ... (文件下载和清理逻辑保持不变) ...
            if final_result_path and os.path.exists(final_result_path) and not processing_has_failed:
                st.download_button(
                    label=f"下载结果 ({os.path.basename(final_result_path)})",
                    data=full_llm_response,
                    file_name=os.path.basename(final_result_path),
                    mime="text/markdown",
                    use_container_width=True
                )
            
            if not keep_temp_files:
                try:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                except OSError as e:
                    st.warning(f"无法自动删除临时上传文件 '{temp_file_path}': {e}")

                transcript_path = "source_transcript.txt"
                try:
                    if os.path.exists(transcript_path):
                        os.remove(transcript_path)
                except OSError as e:
                    st.warning(f"无法自动删除文字稿文件 '{transcript_path}': {e}")
            else:
                st.info("已根据您的设置，保留了中间文件。")