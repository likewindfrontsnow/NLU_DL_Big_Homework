# app.py
import streamlit as st
import os
from main import main_process_generator
from config import LLM_CONFIG 
# (新增) 导入精炼函数
from llm_api import refine_llm_generation 

st.set_page_config(page_title="智能笔记 Agent", layout="wide")
st.title("👨‍💻 智能内容生成 Agent")
# (TA 修改) 移除 Q&A 和 Quiz
st.markdown("上传您的视频、音频或文本文档，即可自动生成结构化笔记。")

# --- (新增) Session State 初始化 ---
# 用于在重新运行 (rerun) 之间保持状态
if "processing_started" not in st.session_state:
    st.session_state.processing_started = False
if "current_notes" not in st.session_state:
    st.session_state.current_notes = None # 存储生成的笔记
if "full_transcript" not in st.session_state:
    st.session_state.full_transcript = None # 存储原始转录稿
if "output_filename" not in st.session_state:
    st.session_state.output_filename = "我的学习笔记"
if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None
if "processing_has_failed" not in st.session_state:
    st.session_state.processing_has_failed = False
# (TA 修改) 为 Qwen ASR 上下文添加 session state
if "asr_context" not in st.session_state:
    st.session_state.asr_context = ""
# --- 结束新增 ---

provider_name = LLM_CONFIG.get('provider_name', 'LLM')
st.info(f"💡 **提示**: 视频/音频文件将使用所选转录服务，笔记生成将调用 **{provider_name}** API。")

with st.sidebar:
    st.header("⚙️ 参数配置")

    st.session_state.output_filename = st.text_input(
        "请输入希望的笔记文件名 (无需后缀)", 
        value=st.session_state.output_filename
    )

    # (TA 修改) 完整移除了 query_option (Q&A / Quiz) 的选择框
    
    st.markdown("---")
    st.subheader("语音转录 (ASR) 配置")
    transcription_provider = st.radio(
        "请选择语音转录服务:",
        ("Local Whisper", "Qwen API"),
        index=0,
        key="transcription_provider",
        help="""
        - **Local Whisper**: 在您本地电脑上运行，速度取决于您的电脑配置，无需联网。
        - **Qwen API**: 调用阿里云 Qwen ASR API，速度快，精度可能更高，但需要联网且音频文件会被上传。
        """
    )
    
    whisper_model_size = "tiny" 
    if transcription_provider == "Local Whisper":
        whisper_model_size = st.selectbox(
            "请选择 Whisper 模型:",
            ("tiny", "base", "small", "medium", "large"),
            index=0,
            help="模型越大，转录越准确，但速度越慢。'tiny' 最快，'large' 最准。首次使用非 'tiny' 模型时，程序会先下载模型文件（可能需要几分钟）。"
        )
        # (TA 修改) 如果没选 Qwen，清空上下文
        if st.session_state.asr_context != "":
            st.session_state.asr_context = "" 
    else:
        st.info("Qwen API 将使用 qwen3-asr-flash 模型。")
    
    # (TA 修改) 仅在选择 Qwen API 时显示上下文输入框
    if transcription_provider == "Qwen API":
        st.markdown("---")
        st.subheader("ASR 上下文增强 (Qwen)")
        # (TA 修改) 使用 session_state 绑定
        st.session_state.asr_context = st.text_area(
            "输入专业术语 (用于提升 ASR 准确率)",
            value=st.session_state.asr_context,
            placeholder="例如: Bulge Bracket, Boutique, 投行...",
            help="在此处输入希望 Qwen API 优先识别的专业词汇、人名或地名，用逗号或段落分隔均可。仅在选择 Qwen API 时生效。"
        )


    st.markdown("---")
    stream_output = st.toggle(
        "启用笔记流式输出", 
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

# --- (修改) 检查是否上传了新文件 ---
if uploaded_file is not None and st.session_state.last_uploaded_filename != uploaded_file.name:
    # 如果是新文件，重置所有状态
    st.session_state.processing_started = False
    st.session_state.current_notes = None
    st.session_state.full_transcript = None
    st.session_state.last_uploaded_filename = uploaded_file.name
    st.session_state.processing_has_failed = False
    # (重要) 立即刷新，让 "开始生成" 按钮出现
    st.rerun() # <-- (修正)


# --- (修改) 逻辑块 1: 仅在“未开始”且“未失败”时显示“开始”按钮 ---
if uploaded_file is not None and not st.session_state.processing_started and not st.session_state.processing_has_failed:
    if st.button("开始生成", use_container_width=True, type="primary"):
        # 标记处理已开始，并立即刷新
        st.session_state.processing_started = True
        st.rerun() # <-- (修正)

# --- (修改) 逻辑块 2: 仅在“已开始”但“笔记未生成”时运行处理 ---
if st.session_state.processing_started and st.session_state.current_notes is None and not st.session_state.processing_has_failed:
    
    st.markdown("---")
    st.subheader("处理进度")
    
    main_progress_bar = st.progress(0)
    main_progress_text = st.empty()
    sub_progress_bar = st.progress(0)
    sub_progress_text = st.empty()

    st.markdown("---")

    stream_status = "流式" if stream_output else "非流式"
    # (TA 修改) 移除了 processing_headers 字典，直接使用固定标题
    st.subheader(f"正在生成笔记 ({provider_name} {stream_status})...")
    
    if transcription_provider == "Local Whisper":
        asr_config_text = f"转录服务: **Local Whisper** (模型: **{whisper_model_size}**)"
    else:
        asr_config_text = "转录服务: **Qwen API** (模型: **qwen3-asr-flash**)"
        # (TA 修改) 如果有上下文，也显示出来
        if st.session_state.asr_context:
            asr_config_text += f" | **上下文:** *{st.session_state.asr_context[:30]}...*"
    
    # (TA 修改) 移除了 query_option
    st.info(f"{asr_config_text}")
    
    llm_output_container = st.empty()
    full_llm_response = ""
    
    final_result_path = None
    
    # 重新获取文件路径 (因为 state 刷新了)
    temp_dir = "temp_uploads"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    temp_file_path = os.path.join(temp_dir, st.session_state.last_uploaded_filename)
    
    # (重要) 检查文件是否还存在，如果不存在（例如用户清空了缓存），则提示
    try:
        if not os.path.exists(temp_file_path):
            # 重新写入
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
    except Exception as e:
         st.error(f"无法访问临时文件: {e}。请重新上传。")
         st.session_state.processing_started = False
         st.rerun() # <-- (修正)

    # (TA 修改) 移除了 generator 调用的 query_option 参数
    generator = main_process_generator(
        temp_file_path, 
        "DUMMY_KEY", 
        st.session_state.output_filename, 
        whisper_model_size,
        stream_output,
        transcription_provider,
        st.session_state.asr_context # (TA 修改) 传入上下文参数
    )
    
    for event_type, value, *rest in generator:
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
        
        elif event_type == "persistent_error":
            st.error(f"处理失败: {text}")
            main_progress_text.error("一个关键步骤在多次重试后仍然失败，已停止处理。")
            llm_output_container.error(f"**错误详情:**\n\n{text}")
            st.session_state.processing_has_failed = True # 标记失败
            st.rerun() # <-- (修正)
            break
        
        elif event_type == "error":
            st.error(text)
            llm_output_container.error(text)
            st.session_state.processing_has_failed = True # 标记失败
            st.rerun() # <-- (修正)
            break

        elif event_type == "done":
            main_progress_bar.progress(1.0)
            sub_progress_bar.empty()
            sub_progress_text.empty()
            
            # (修改) 不再在此处显示，只保存到 state
            # llm_output_container.markdown(full_llm_response) 
            st.success(text)
            final_result_path = value
            
            st.session_state.current_notes = full_llm_response
            
            # (修改) 清理临时文件
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
            
            # (关键) 处理完成，立即刷新
            st.rerun() # <-- (修正)
            break


# --- (修改) 逻辑块 3: 仅在“笔记已生成”时显示笔记和精炼 UI ---
if st.session_state.current_notes:
    
    st.markdown("---")
    st.subheader("🎉 智能笔记")
    
    # (修改) 这是笔记的唯一显示区域
    note_display_area = st.empty()
    note_display_area.markdown(st.session_state.current_notes)
        
    st.download_button(
        label=f"下载当前笔记 ({st.session_state.output_filename}.md)",
        data=st.session_state.current_notes,
        file_name=f"{st.session_state.output_filename}.md",
        mime="text/markdown",
        use_container_width=True
    )
    
    st.markdown("---")
    st.subheader("✍️ 笔记精炼")
    st.info("对当前生成的笔记不满意？请选择快捷指令或输入您的修改意见。")

    col1, col2 = st.columns(2)
    
    with col1:
        preset_feedback = st.selectbox(
            "快捷指令:",
            (
                "(请选择一个快捷指令)", 
                "帮我总结得更简洁", 
                "帮我扩写得更详细 (需要参考原始转录稿)", 
                "把语气变得更生动有趣",
                "把语气变得更专业严肃",
                "帮我用项目符号(bullet points)重新组织"
            )
        )
    
    with col2:
        custom_feedback = st.text_input(
            "或输入你的自定义指令:", 
            placeholder="例如：请重点扩写第二部分..."
        )

    if st.button("🚀 开始精炼", use_container_width=True, type="primary"):
        feedback = custom_feedback if custom_feedback else preset_feedback
        
        if feedback == "(请选择一个快捷指令)" or not feedback:
            st.warning("请输入或选择一个修改指令。")
        elif not st.session_state.full_transcript:
            st.error("错误：未找到原始转录稿，无法进行精炼。请重新处理文件。")
        else:
            st.info("正在根据您的反馈重新生成笔记...")
            # (修改) 直接在唯一的显示区域流式输出
            refined_notes = ""
            
            try:
                regenerator = refine_llm_generation(
                    original_transcript=st.session_state.full_transcript,
                    current_notes=st.session_state.current_notes,
                    user_feedback=feedback,
                    stream_output=stream_output
                )

                if stream_output:
                    for chunk in regenerator:
                        if chunk:
                            refined_notes += chunk
                            note_display_area.markdown(refined_notes) # 实时替换
                else:
                    refined_notes = regenerator
                    note_display_area.markdown(refined_notes)
                
                # (关键) 用新笔记覆盖旧笔记
                st.session_state.current_notes = refined_notes
                
                # (新增) 保存精炼后的笔记
                try:
                    save_path = f"{st.session_state.output_filename}_refined.md"
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(refined_notes)
                    st.success(f"精炼完成！")
                except IOError as e:
                    st.error(f"保存精炼笔记失败: {e}")
                
                # (新增) 再次 Rerun 以清理“正在精炼”的提示
                st.rerun() # <-- (修正)

            except Exception as e:
                st.error(f"精炼过程中出错: {e}")

# --- (修改) 逻辑块 4: 仅在“失败”时显示重试按钮 ---
if st.session_state.processing_has_failed:
    st.error("上次处理失败。请检查文件或配置。")
    if st.button("🔄 重新开始", use_container_width=True):
        # 重置所有状态
        st.session_state.processing_started = False
        st.session_state.current_notes = None
        st.session_state.full_transcript = None
        st.session_state.last_uploaded_filename = None
        st.session_state.processing_has_failed = False
        st.rerun() # <-- (修正)