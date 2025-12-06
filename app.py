import streamlit as st
import os
import time
import shutil
import atexit
import signal
import sys
from main import main_process_generator
from core.config import LLM_CONFIG, save_models_config, load_models_config, ASR_MODELS_WITH_CONTEXT_SUPPORT
from ai_services.llm_api import refine_llm_generation
from dotenv import set_key
from core.model_verifier import verify_llm_model, verify_asr_model, verify_vlm_model

VIDEO_EXTS = {'mp4', 'mov', 'mpeg', 'webm'}
AUDIO_EXTS = {'mp3', 'm4a', 'wav', 'amr', 'mpga'}
DOC_EXTS = {'txt', 'md', 'mdx', 'markdown', 'pdf', 'html', 'xlsx', 'xls', 'doc', 'docx', 'csv', 'eml', 'msg', 'pptx', 'ppt', 'xml', 'epub'}
ALL_EXTS = list(VIDEO_EXTS | AUDIO_EXTS | DOC_EXTS)

INSTRUCTION_OPTIONS = [
    "请尤其注意老师提到的与考试相关的部分",
    "请将所有专业术语中英文对照列出",
    "请使用更多的表格来对比易混淆的概念",
    "语气要更加幽默风趣，像个老朋友在聊天",
    "只保留核心考点，极度精简，不要废话",
    "为每个章节添加 emoji 图标，增加可读性",
    "请详细解释所有缩写词 (Abbreviations)"
]

NOTE_TYPE_MAPPING = {
    "STEM": "理工科",
    "HASS": "人文社科",
    "Medical": "医学"
}

def global_cleanup():
    temp_dir = "temp"
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
    except Exception:
        pass

atexit.register(global_cleanup)

def signal_handler(sig, frame):
    global_cleanup()
    sys.exit(0)

if sys.platform != 'win32':
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    except:
        pass

st.set_page_config(page_title="智能笔记 Agent", layout="wide")
st.title("👨‍💻 智能内容生成 Agent")

def cleanup_temp_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass

def _save_key(key):
    env_path = ".env"
    set_key(env_path, "LLM_API_KEY", key)
    set_key(env_path, "DASHSCOPE_API_KEY", key)
    os.environ["LLM_API_KEY"] = key
    os.environ["DASHSCOPE_API_KEY"] = key
    LLM_CONFIG["api_key"] = key
    if not LLM_CONFIG.get("provider_name"):
        LLM_CONFIG["provider_name"] = "DashCope(Qwen)"
        set_key(env_path, "LLM_PROVIDER_NAME", "DashCope(Qwen)")

def init_model_config_state():
    if "llm_models" not in st.session_state:
        config = load_models_config()
        st.session_state.llm_models = config.get("llm_models", [])
        st.session_state.vlm_models = config.get("vlm_models", [])
        st.session_state.asr_models = config.get("asr_models", [])
        
        LLM_CONFIG["supported_llm"] = st.session_state.llm_models
        LLM_CONFIG["supported_vlm"] = st.session_state.vlm_models
        LLM_CONFIG["supported_asr_models"] = st.session_state.asr_models

init_model_config_state()

if "has_validated_on_startup" not in st.session_state:
    st.session_state.has_validated_on_startup = False

if LLM_CONFIG.get("api_key") and not st.session_state.has_validated_on_startup:
    with st.spinner("🔄 正在启动自检：验证 API Key 有效性..."):
        check_key = LLM_CONFIG["api_key"]
        check_url = LLM_CONFIG.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        check_model = LLM_CONFIG.get("model", st.session_state.llm_models[0] if st.session_state.llm_models else "qwen-plus")
        
        is_valid, msg = verify_llm_model(check_key, check_url, check_model)
        
        if is_valid:
            st.session_state.has_validated_on_startup = True
        else:
            st.error(f"⚠️ 启动检测警告：当前保存的 API Key 无效或过期。\n\n**原因**: {msg}")
            st.warning("请在下方重新输入有效的 API Key。")
            LLM_CONFIG["api_key"] = ""

if not LLM_CONFIG.get("api_key"):
    if not st.session_state.get("has_validated_on_startup", False):
         st.markdown("### 👋 欢迎使用")
    
    st.markdown("""
    请输入您的 DashScope (Qwen) API Key 以开始使用。
    
    - Key 将被安全地保存到本地 `.env` 文件中。
    - 您可以前往 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/apiKey) 获取您的 Key。
    """)
    
    new_key = st.text_input(
        "请输入您的 Qwen API Key (sk-xxxxxxxx)", 
        type="password", 
        key="api_key_input_main"
    )
    
    if st.button("验证并保存", use_container_width=True, type="primary"):
        if new_key and new_key.startswith("sk-"):
            st.info("正在连接服务器验证 Key 的有效性...")
            
            verify_url = LLM_CONFIG.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            verify_model = st.session_state.llm_models[0] if st.session_state.llm_models else "qwen-plus"
            
            is_valid, msg = verify_llm_model(new_key, verify_url, verify_model)

            if is_valid:
                _save_key(new_key)
                st.session_state.has_validated_on_startup = True
                st.success("✅ API Key 验证通过并已保存！")
                time.sleep(1)
                st.rerun()
            else:
                st.error(f"❌ 验证失败：{msg}")
                st.markdown("👉 **请检查您的 Key 是否正确，或网络是否通畅。**")
        else:
            st.error("Key 格式不正确。它应该以 `sk-` 开头。")

else:
    st.markdown("上传您的视频、音频或文本文档，即可自动生成结构化笔记。")

    default_states = {
        "processing_started": False,
        "current_notes": None,
        "full_transcript": None,
        "output_filename": "我的学习笔记",
        "last_uploaded_filename": None,
        "processing_has_failed": False,
        "api_key_invalid": False,
        "asr_context": "",
        "note_type": "STEM",
        "stop_requested": False,
        "refinement_in_progress": False,
        "refinement_stop_requested": False,
        "preset_feedback": "(请选择一个快捷指令)",
        "custom_feedback": "",
    }

    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

    is_busy = st.session_state.processing_started or st.session_state.refinement_in_progress

    provider_name = LLM_CONFIG.get('provider_name', 'LLM')
    st.info(f"💡 **提示**: 视频/音频文件将使用所选转录服务，笔记生成将调用 **{provider_name}** API。")

    with st.sidebar:
        st.header("⚙️ 参数配置")

        current_llm_model = LLM_CONFIG.get("model")
        available_llm = st.session_state.llm_models
        
        if current_llm_model not in available_llm and available_llm:
             current_llm_model = available_llm[0]
        elif not current_llm_model and available_llm:
             current_llm_model = available_llm[0]
        
        try:
            llm_index = available_llm.index(current_llm_model) if current_llm_model in available_llm else 0
        except ValueError:
            llm_index = 0
            
        selected_llm_model = st.selectbox(
            "请选择笔记生成模型 (LLM):",
            available_llm,
            index=llm_index,
            disabled=is_busy,
            help="选择用于生成笔记的大语言模型。"
        )
        LLM_CONFIG["model"] = selected_llm_model
        set_key(".env", "LLM_MODEL", selected_llm_model)

        st.session_state.output_filename = st.text_input(
            "请输入希望的笔记文件名 (无需后缀)", 
            value=st.session_state.output_filename,
            disabled=is_busy
        )

        note_type_option = st.radio(
            "请选择笔记类型:",
            ("STEM", "HASS","Medical"),
            index=0, 
            key="note_type", 
            horizontal=True,
            format_func=lambda x: NOTE_TYPE_MAPPING.get(x, x),
            disabled=is_busy
        )

        st.markdown("---")
        st.subheader("语音转录 (ASR) 配置")
        transcription_provider = st.radio(
            "请选择语音转录服务:",
            ("Qwen API","Local Whisper"),
            index=0,
            key="transcription_provider",
            help="选择转录服务。",
            disabled=is_busy
        )
        
        whisper_model_size = "tiny" 
        qwen_asr_model = "qwen-audio-asr-latest"
        selected_backup = None
        asr_workers = 10

        if transcription_provider == "Local Whisper":
            whisper_model_size = st.selectbox(
                "请选择 Whisper 模型:",
                ("tiny", "base", "small", "medium", "large"),
                index=0,
                help="模型越大，转录越准确，但速度越慢。",
                disabled=is_busy
            )
            if st.session_state.asr_context != "":
                st.session_state.asr_context = "" 
        else:
            qwen_asr_model_options = st.session_state.asr_models
            if not qwen_asr_model_options:
                 qwen_asr_model_options = ["qwen3-asr-flash"]

            qwen_asr_model_index = 0
            if "qwen3-asr-flash" in qwen_asr_model_options:
                qwen_asr_model_index = qwen_asr_model_options.index("qwen3-asr-flash")

            qwen_asr_model = st.selectbox(
                "请选择 Qwen ASR 模型:",
                qwen_asr_model_options,
                index=qwen_asr_model_index,
                disabled=is_busy,
                help="选择用于语音转录的模型。"
            )
            
            backup_options = ["(无备选)"] + qwen_asr_model_options
            selected_backup_str = st.selectbox(
                "请选择备选模型 (遇到限流时自动切换):",
                backup_options,
                index=0,
                disabled=is_busy
            )
            selected_backup = None if selected_backup_str == "(无备选)" else selected_backup_str
            LLM_CONFIG["asr_backup_model"] = selected_backup

            asr_workers = st.slider(
                "ASR 并发线程数",
                min_value=1,
                max_value=20,
                value=10,
                help="增加线程数可加快转录速度，但可能触发 API 限流。",
                disabled=is_busy
            )
        
        if transcription_provider == "Qwen API":
            if qwen_asr_model in ASR_MODELS_WITH_CONTEXT_SUPPORT:
                st.markdown("---")
                st.subheader("ASR 上下文增强 (Qwen)")
                st.session_state.asr_context = st.text_area(
                    "输入热词 (用于提升 ASR 准确率)",
                    value=st.session_state.asr_context,
                    placeholder="例如: Bulge Bracket, Boutique, 投行...",
                    help="在此处输入希望 Qwen API 优先识别的专业词汇、人名或地名。",
                    disabled=is_busy
                )
            else:
                if st.session_state.asr_context != "":
                     st.session_state.asr_context = ""

        st.markdown("---")
        
        stream_output = st.toggle(
            "启用笔记流式输出", 
            value=True, 
            help="启用后，笔记内容将实时逐字显示。",
            disabled=is_busy
        )

        enable_visual_analysis = st.toggle(
            "启用视频视觉分析 (VLM)",
            value=False,
            help="开启后，将对视频进行抽帧分析，提取PPT和板书内容。这会增加处理时间。",
            disabled=is_busy
        )
        
        selected_vlm_model = "qwen3-vl-plus"
        selected_vlm_backup = None
        vlm_workers = 10
        keep_visual_files = False
        insert_images = False

        if enable_visual_analysis:
            selected_vlm_model_options = st.session_state.vlm_models
            if not selected_vlm_model_options:
                 selected_vlm_model_options = ["qwen3-vl-plus"]

            selected_vlm_model = st.selectbox(
                "请选择 VLM 模型:",
                selected_vlm_model_options,
                index=0,
                disabled=is_busy
            )

            vlm_backup_options = ["(无备选)"] + selected_vlm_model_options
            selected_vlm_backup_str = st.selectbox(
                "请选择 VLM 备选模型:",
                vlm_backup_options,
                index=0,
                disabled=is_busy
            )
            selected_vlm_backup = None if selected_vlm_backup_str == "(无备选)" else selected_vlm_backup_str
            LLM_CONFIG["vlm_backup_model"] = selected_vlm_backup

            vlm_workers = st.slider(
                "VLM 并发线程数",
                min_value=1,
                max_value=20,
                value=10,
                help="并发数越高速度越快，但容易触发 API 限流。",
                disabled=is_busy
            )

            keep_visual_files = st.checkbox(
                "保留视觉分析中间文件 (截图 & 报告)",
                value=False,
                disabled=is_busy,
                help="勾选后，抽取的关键帧和生成的视觉报告将不会被自动删除。"
            )
            insert_images = st.checkbox(
                "智能插入图片到笔记",
                value=False,
                disabled=is_busy,
                help="勾选后，生成的笔记将包含视频关键帧的图片链接。"
            )

        st.markdown("---")

        if st.button("🔍 全面检测模型连通性", disabled=is_busy, use_container_width=True):
            current_api_key = os.getenv("DASHSCOPE_API_KEY") or LLM_CONFIG.get("api_key")
            current_base_url = LLM_CONFIG.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            
            if not current_api_key:
                st.error("❌ 未检测到 API Key，请先在上一步输入并保存。")
            else:
                with st.status("正在进行全链路服务检测...", expanded=True) as status:
                    all_passed = True
                    
                    st.write(f"1. 正在连接笔记生成 LLM: **{selected_llm_model}** ...")
                    llm_ok, llm_msg = verify_llm_model(current_api_key, current_base_url, selected_llm_model)
                    if llm_ok:
                        st.write(f":green[{llm_msg}]")
                    else:
                        st.write(f":red[{llm_msg}]")
                        all_passed = False

                    if transcription_provider == "Qwen API":
                        st.write(f"2. 正在连接 ASR 主模型: **{qwen_asr_model}** ...")
                        asr_ok, asr_msg = verify_asr_model(current_api_key, qwen_asr_model)
                        if asr_ok:
                            st.write(f":green[{asr_msg}]")
                        else:
                            st.write(f":red[{asr_msg}]")
                            all_passed = False
                        
                        if selected_backup:
                            st.write(f"3. 正在连接 ASR 备选模型: **{selected_backup}** ...")
                            bk_ok, bk_msg = verify_asr_model(current_api_key, selected_backup)
                            if bk_ok:
                                st.write(f":green[{bk_msg}]")
                            else:
                                st.write(f":red[{bk_msg}]")
                                all_passed = False
                    else:
                        st.write("2. ASR 服务选定为 **Local Whisper**，跳过云端验证。")

                    if enable_visual_analysis:
                        st.write(f"4. 正在连接视觉分析 VLM: **{selected_vlm_model}** ...")
                        vlm_ok, vlm_msg = verify_vlm_model(current_api_key, selected_vlm_model)
                        if vlm_ok:
                            st.write(f":green[{vlm_msg}]")
                        else:
                            st.write(f":red[{vlm_msg}]")
                            all_passed = False
                        
                        if selected_vlm_backup:
                            st.write(f"5. 正在连接 VLM 备选模型: **{selected_vlm_backup}** ...")
                            vb_ok, vb_msg = verify_vlm_model(current_api_key, selected_vlm_backup)
                            if vb_ok:
                                st.write(f":green[{vb_msg}]")
                            else:
                                st.write(f":red[{vb_msg}]")
                                all_passed = False

                    if all_passed:
                        status.update(label="✅ 所有选定服务连接正常！", state="complete", expanded=False)
                        st.toast("✅ 模型连通性检测通过！", icon="🎉")
                    else:
                        status.update(label="❌ 检测到服务连接问题", state="error", expanded=True)
                        st.error("请检查报错信息，确认 API Key 余额或模型名称是否正确。")

        st.markdown("---")
        keep_temp_files = st.checkbox(
            "保留语音转文字稿", 
            value=False, 
            help="勾选后将保留语音转文字生成的 .txt 文字稿。",
            disabled=is_busy
        )

        st.markdown("---")
        with st.expander("🎨 个性化定制 (生成前)", expanded=False):
            st.markdown("在这里添加对笔记生成的特殊要求。")
            
            selected_instructions = st.multiselect(
                "快捷指令 (多选):",
                INSTRUCTION_OPTIONS,
                key="selected_instructions_ui",
                disabled=is_busy
            )
            
            custom_instruction_text = st.text_area(
                "自定义额外要求:",
                placeholder="例如：请重点关注药理学部分，特别是副作用...",
                key="custom_instruction_text_ui",
                disabled=is_busy
            )

        final_custom_instructions = ""
        if selected_instructions:
            final_custom_instructions += "、".join(selected_instructions) + "。\n"
        if custom_instruction_text:
            final_custom_instructions += custom_instruction_text
        
        st.markdown("---")

        with st.expander("🛠️ 模型管理 (列表配置)", expanded=False):
            st.info("在此添加或删除支持的模型。修改将即时生效。")
            
            tab_llm, tab_vlm, tab_asr = st.tabs(["LLM (笔记)", "VLM (视觉)", "ASR (语音)"])
            
            def render_model_manager(tab_obj, model_list_key, label):
                with tab_obj:
                    current_list = st.session_state[model_list_key]
                    st.write(f"**当前可用 {label} 模型列表:**")
                    
                    st.markdown(
                        f"""
                        <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px;">
                            {''.join([f'<span style="background-color: #f0f2f6; padding: 4px 8px; border-radius: 4px; border: 1px solid #e0e0e0;">{m}</span>' for m in current_list])}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )

                    col_add, col_del = st.columns(2)
                    
                    with col_add:
                        st.markdown("##### ➕ 添加模型")
                        new_model_name = st.text_input(f"输入新模型名称 ({label})", key=f"new_{model_list_key}")
                        if st.button(f"添加至 {label} 列表", key=f"btn_add_{model_list_key}", disabled=is_busy):
                            if new_model_name and new_model_name.strip():
                                clean_name = new_model_name.strip()
                                if clean_name not in st.session_state[model_list_key]:
                                    st.session_state[model_list_key].append(clean_name)
                                    new_config = {
                                        "llm_models": st.session_state.llm_models,
                                        "vlm_models": st.session_state.vlm_models,
                                        "asr_models": st.session_state.asr_models
                                    }
                                    save_models_config(new_config)
                                    st.success(f"已添加 {clean_name}")
                                    time.sleep(0.5)
                                    st.rerun()
                                else:
                                    st.warning("该模型已存在。")
                            else:
                                st.warning("请输入模型名称。")

                    with col_del:
                        st.markdown("##### 🗑️ 删除模型")
                        models_to_delete = st.multiselect(
                            f"选择要删除的模型 ({label})", 
                            st.session_state[model_list_key],
                            key=f"del_select_{model_list_key}"
                        )
                        if st.button(f"确认删除 ({len(models_to_delete)})", key=f"btn_del_{model_list_key}", disabled=is_busy):
                            if models_to_delete:
                                for m in models_to_delete:
                                    if m in st.session_state[model_list_key]:
                                        st.session_state[model_list_key].remove(m)
                                new_config = {
                                    "llm_models": st.session_state.llm_models,
                                    "vlm_models": st.session_state.vlm_models,
                                    "asr_models": st.session_state.asr_models
                                }
                                save_models_config(new_config)
                                st.success("已删除选中模型")
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.warning("请先选择要删除的模型。")

            render_model_manager(tab_llm, "llm_models", "LLM")
            render_model_manager(tab_vlm, "vlm_models", "VLM")
            render_model_manager(tab_asr, "asr_models", "ASR")

        st.info("请在上方配置好参数后，上传文件开始处理。")

    with st.expander("查看所有支持的文件格式"):
        st.markdown(f"""
        - **视频文件**: `{', '.join(sorted(list(VIDEO_EXTS)))}`
        - **音频文件**: `{', '.join(sorted(list(AUDIO_EXTS)))}`
        - **文档文件**: `{', '.join(sorted(list(DOC_EXTS)))}`
        """)

    uploaded_file = st.file_uploader(
        "上传视频、音频或文档", 
        type=ALL_EXTS,
        disabled=is_busy
    )

    if uploaded_file is not None and st.session_state.last_uploaded_filename != uploaded_file.name:
        st.session_state.processing_started = False
        st.session_state.current_notes = None
        st.session_state.full_transcript = None
        st.session_state.last_uploaded_filename = uploaded_file.name
        st.session_state.processing_has_failed = False
        st.rerun() 

    if uploaded_file is not None and not is_busy and not st.session_state.processing_has_failed:
        if st.button("开始生成", use_container_width=True, type="primary"):
            st.session_state.processing_started = True
            st.session_state.current_notes = None       
            st.session_state.full_transcript = None     
            st.session_state.processing_has_failed = False
            st.session_state.stop_requested = False
            st.session_state.refinement_in_progress = False
            st.session_state.refinement_stop_requested = False
            st.rerun()

    def handle_stop():
        st.session_state.stop_requested = True

    def on_preset_change():
        if st.session_state.preset_feedback != "(请选择一个快捷指令)":
            st.session_state.custom_feedback = ""

    def on_custom_change():
        if st.session_state.custom_feedback != "":
            st.session_state.preset_feedback = "(请选择一个快捷指令)"


    stop_button_placeholder = st.empty()
    temp_file_path = None
    
    if st.session_state.processing_started and st.session_state.current_notes is None and not st.session_state.processing_has_failed:
        
        stop_button_placeholder.button("⏹️ 停止生成", on_click=handle_stop, use_container_width=True)
        
        st.markdown("---")
        st.subheader("处理进度")
        
        main_progress_bar = st.progress(0)
        main_progress_text = st.empty()
        sub_progress_bar = st.progress(0)
        sub_progress_text = st.empty()

        st.markdown("---")

        stream_status = "流式" if stream_output else "非流式"
        st.subheader(f"正在生成笔记 ({provider_name} / {selected_llm_model} {stream_status})...")
        
        if transcription_provider == "Local Whisper":
            asr_config_text = f"转录服务: **Local Whisper** (模型: **{whisper_model_size}**)"
        else:
            asr_config_text = f"转录服务: **Qwen API** (模型: **{qwen_asr_model}**)"
            if st.session_state.asr_context:
                asr_config_text += f" | **上下文:** *{st.session_state.asr_context[:30]}...*"
        
        st.info(f"{asr_config_text} | 笔记类型: **{st.session_state.note_type}**")
        
        llm_output_container = st.empty()
        full_llm_response = ""
        
        final_result_path = None
        
        temp_root = "temp"
        temp_dir = os.path.join(temp_root, "uploads")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_file_path = os.path.join(temp_dir, st.session_state.last_uploaded_filename)
        
        try:
            if not os.path.exists(temp_file_path):
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
        except Exception as e:
             st.error(f"无法访问临时文件: {e}。请重新上传。")
             st.session_state.processing_started = False
             st.rerun() 

        generator = main_process_generator(
            temp_file_path, 
            st.session_state.output_filename, 
            whisper_model_size,
            stream_output,
            transcription_provider,
            st.session_state.note_type, 
            st.session_state.asr_context,
            final_custom_instructions,
            qwen_asr_model=qwen_asr_model,
            enable_visual_analysis=enable_visual_analysis,
            vlm_model_name=selected_vlm_model,
            vlm_backup_model=selected_vlm_backup,
            keep_visual_files=keep_visual_files,
            insert_images=insert_images,
            vlm_concurrency=vlm_workers,
            asr_concurrency=asr_workers
        )
        
        try:
            for event_type, value, *rest in generator:
                
                if st.session_state.get("stop_requested", False):
                    st.session_state.processing_started = False
                    stop_button_placeholder.empty()
                    st.warning("处理已由用户手动停止。")
                    main_progress_text.warning("处理已停止。")
                    generator.close()
                    break

                text = rest[0] if rest else ""

                if event_type == "progress":
                    main_progress_bar.progress(float(value))
                    main_progress_text.info(text)
                elif event_type == "sub_progress":
                    sub_progress_bar.progress(float(value))
                    sub_progress_text.text(text)
                
                elif event_type == "transcript":
                    st.session_state.full_transcript = value

                elif event_type == "llm_chunk":
                    full_llm_response += value
                    llm_output_container.markdown(full_llm_response) 
                
                elif event_type == "persistent_error" or event_type == "error":
                    st.error(f"处理失败: {text}")
                    main_progress_text.error("一个关键步骤在多次重试后仍然失败，已停止处理。")
                    llm_output_container.error(f"**错误详情:**\n\n{text}")
                    
                    if "401" in text or "密钥无效" in text or "Unauthorized" in text:
                        st.session_state.api_key_invalid = True
                    cleanup_temp_file(temp_file_path)

                    st.session_state.processing_has_failed = True 
                    st.session_state.processing_started = False
                    stop_button_placeholder.empty()
                    st.rerun() 
                    break

                elif event_type == "done":
                    main_progress_bar.progress(1.0)
                    sub_progress_bar.empty()
                    sub_progress_text.empty()
                    
                    st.success(text)
                    
                    st.session_state.current_notes = full_llm_response
                    st.session_state.processing_started = False
                    stop_button_placeholder.empty()
                    
                    if not keep_temp_files:
                        transcript_path = os.path.join("temp", "source_transcript.txt")
                        try:
                            if os.path.exists(transcript_path):
                                os.remove(transcript_path)
                        except OSError as e:
                            st.warning(f"无法自动删除文字稿文件 '{transcript_path}': {e}")
                    else:
                        st.info("已根据您的设置，保留了语音转文字稿。")
                    
                    cleanup_temp_file(temp_file_path)
                    st.rerun() 
        finally:
            if st.session_state.processing_started:
                 st.session_state.processing_started = False
                 stop_button_placeholder.empty()
                 st.rerun()
            elif st.session_state.get("stop_requested", False):
                 st.session_state.stop_requested = False
                 st.session_state.processing_started = False
                 stop_button_placeholder.empty()
                 st.rerun()


    if st.session_state.current_notes:
        
        st.markdown("---")
        st.subheader("🎉 智能笔记")
        
        note_display_area = st.empty()
        note_display_area.markdown(st.session_state.current_notes)
            
        st.download_button(
            label=f"下载当前笔记 ({st.session_state.output_filename}.md)",
            data=st.session_state.current_notes,
            file_name=f"{st.session_state.output_filename}.md",
            mime="text/markdown",
            use_container_width=True,
            disabled=is_busy
        )
        
        st.markdown("---")
        st.subheader("✍️ 笔记精炼")
        
        def handle_refinement_stop():
            st.session_state.refinement_stop_requested = True
        
        refinement_stop_button_placeholder = st.empty()
        
        st.info("对当前生成的笔记不满意？请选择快捷指令或输入您的修改意见。")

        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox(
                "快捷指令:",
                (
                    "(请选择一个快捷指令)", 
                    "帮我总结得更简洁", 
                    "帮我扩写得更详细 (需要参考原始转录稿)", 
                    "把语气变得更生动有趣",
                    "把语气变得更专业严肃",
                    "帮我用项目符号(bullet points)重新组织"
                ),
                disabled=is_busy,
                key="preset_feedback",
                on_change=on_preset_change
            )
        
        with col2:
            st.text_input(
                "或输入你的自定义指令:", 
                placeholder="例如：请重点扩写第二部分...",
                disabled=is_busy,
                key="custom_feedback",
                on_change=on_custom_change
            )

        preset_val = st.session_state.preset_feedback
        custom_val = st.session_state.custom_feedback
        feedback = custom_val if custom_val else preset_val
        
        if st.button("🚀 开始精炼", use_container_width=True, type="primary", disabled=is_busy):
            
            if feedback == "(请选择一个快捷指令)" or not feedback:
                st.warning("请输入或选择一个修改指令。")
            elif not st.session_state.full_transcript:
                st.error("错误：未找到原始转录稿，无法进行精炼。请重新处理文件。")
            else:
                st.session_state.refinement_feedback = feedback
                st.session_state.refinement_in_progress = True
                st.session_state.refinement_stop_requested = False
                st.rerun()

    if st.session_state.refinement_in_progress and not st.session_state.processing_started:

        refinement_stop_button_placeholder.button("⏹️ 停止精炼", on_click=handle_refinement_stop, use_container_width=True)
        st.info(f"正在根据指令精炼笔记: *'{st.session_state.refinement_feedback}'*")
        
        refined_notes = ""
        
        try:
            regenerator = refine_llm_generation(
                original_transcript=st.session_state.full_transcript,
                current_notes=st.session_state.current_notes,
                user_feedback=st.session_state.refinement_feedback,
                stream_output=stream_output
            )

            if stream_output:
                for chunk in regenerator:
                    if st.session_state.get("refinement_stop_requested", False):
                        st.warning("精炼已由用户手动停止。")
                        break
                    
                    if chunk:
                        refined_notes += chunk
                        note_display_area.markdown(refined_notes)
            else:
                if not st.session_state.get("refinement_stop_requested", False):
                    refined_notes = regenerator
                else:
                    st.warning("精炼已由用户手动停止。")

            if not st.session_state.get("refinement_stop_requested", False):
                st.session_state.current_notes = refined_notes if refined_notes else st.session_state.current_notes
                try:
                    save_path = os.path.join("output", f"{st.session_state.output_filename}_refined.md")
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(st.session_state.current_notes)
                    st.success(f"精炼完成！")
                except IOError as e:
                    st.error(f"保存精炼笔记失败: {e}")

        except Exception as e:
            st.error(f"精炼过程中出错: {e}")
            if "401" in str(e) or "密钥无效" in str(e):
                st.session_state.api_key_invalid = True
        
        finally:
            st.session_state.refinement_in_progress = False
            st.session_state.refinement_stop_requested = False
            refinement_stop_button_placeholder.empty()
            st.rerun()


    if st.session_state.processing_has_failed:
        st.error("上次处理失败。请检查文件或配置。")
        
        if st.session_state.get("api_key_invalid", False):
            st.error("❌ 检测到您的 API Key 无效或已过期 (401 Unauthorized)。")
            st.info("请输入一个新的 Qwen API Key 并保存。")
            
            new_key_retry = st.text_input(
                "请输入新的 Qwen API Key", 
                type="password", 
                key="api_key_input_retry"
            )
            
            if st.button("保存新 Key", use_container_width=True):
                if new_key_retry and new_key_retry.startswith("sk-"):
                    _save_key(new_key_retry)
                    st.session_state.api_key_invalid = False
                    st.success("新 Key 已保存！请点击下方“重新开始”按钮。")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("Key 格式不正确，请确保以 `sk-` 开头。")

        if st.button("🔄 重新开始", use_container_width=True):
            st.session_state.processing_started = False
            st.session_state.current_notes = None
            st.session_state.full_transcript = None
            st.session_state.last_uploaded_filename = None
            st.session_state.processing_has_failed = False
            st.session_state.api_key_invalid = False
            st.session_state.stop_requested = False
            st.session_state.refinement_in_progress = False
            st.session_state.refinement_stop_requested = False
            st.session_state.preset_feedback = "(请选择一个快捷指令)"
            st.session_state.custom_feedback = ""
            st.rerun()