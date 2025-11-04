# app.py
import streamlit as st
import os
from main import main_process_generator
# (修改) 导入新的通用配置
from config import LLM_CONFIG 

st.set_page_config(page_title="智能笔记 Agent", layout="wide")
st.title("👨‍💻 智能内容生成 Agent")
st.markdown("上传您的视频、音频或文本文档，即可自动生成结构化笔记、Q&A 或测验。")

# --- (修改) 更新提示信息 ---
# (修改) 动态显示 LLM 服务商名称
provider_name = LLM_CONFIG.get('provider_name', 'LLM')
st.info(f"💡 **提示**: 视频/音频文件将使用本地 Whisper 转录，笔记生成将调用 **{provider_name}** API。")

with st.sidebar:
    st.header("⚙️ 参数配置")

    output_filename = st.text_input("请输入希望的笔记文件名 (无需后缀)", value="我的学习笔记")

    query_option = st.selectbox(
        "请选择生成内容类型:",
        ("Notes", "Q&A", "Quiz"),
        index=0,
        help="选择 'Notes' 生成结构化笔记, 'Q&A' 生成问答对, 'Quiz' 生成测验题。(目前仅 'Notes' 功能已接入)"
    )

    # (新) 添加 Whisper 模型大小选择
    whisper_model_size = st.selectbox(
        "请选择语音转录模型:",
        ("tiny", "base", "small", "medium", "large"),
        index=0,
        help="模型越大，转录越准确，但速度越慢。'tiny' 最快，'large' 最准。首次使用非 'tiny' 模型时，程序会先下载模型文件（可能需要几分钟）。"
    )

    # --- (新增) 添加流式输出开关 ---
    stream_output = st.toggle(
        "启用流式输出", 
        value=True, 
        help="启用后，笔记内容将实时逐字显示。禁用则会在所有内容生成后一次性显示。"
    )
    # --- 结束新增 ---

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

if uploaded_file is not None:
    if st.button("开始生成", use_container_width=True, type="primary"):

        st.markdown("---")
        st.subheader("处理进度")
        
        main_progress_bar = st.progress(0)
        main_progress_text = st.empty()
        sub_progress_bar = st.progress(0)
        sub_progress_text = st.empty()

        st.markdown("---")

        # --- (修改) 动态显示流式状态 ---
        stream_status = "流式" if stream_output else "非流式"
        processing_headers = {
            "Notes": f"正在生成笔记 ({provider_name} {stream_status})...",
            "Q&A": "正在进行 Q&A...",
            "Quiz": "正在生成测验..."
        }
        st.subheader(processing_headers.get(query_option, "正在处理..."))
        st.info(f"当前生成模式: **{query_option}** (转录模型: **{whisper_model_size}**)")
        # --- 结束修改 ---
        
        classification_display = st.empty() # (保留占位，但不再使用)
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

        # (修改) 将 whisper_model_size 和 stream_output 传递给主进程
        generator = main_process_generator(
            temp_file_path, 
            "DUMMY_KEY", 
            output_filename, 
            query_option, 
            whisper_model_size,
            stream_output # <-- 新增参数
        )
        
        for event_type, value, *rest in generator:
            text = rest[0] if rest else ""

            if event_type == "progress":
                main_progress_bar.progress(float(value))
                main_progress_text.info(text)
            elif event_type == "sub_progress":
                sub_progress_bar.progress(float(value))
                sub_progress_text.text(text)
            
            # --- (移除) 不再处理 'display_classification' 事件 ---

            elif event_type == "llm_chunk":
                # (修改) 此循环现在可以正确处理流式（多个小块）和非流式（一个大块）
                full_llm_response += value
                llm_output_container.markdown(full_llm_response) # 实时更新
            
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
        
        if final_result_path and os.path.exists(final_result_path) and not processing_has_failed:
            st.download_button(
                label=f"下载结果 ({os.path.basename(final_result_path)})",
                data=full_llm_response,
                file_name=os.path.basename(final_result_path),
                mime="text/markdown",
                use_container_width=True
            )
        
        if not keep_temp_files:
            # ... (文件清理逻辑保持不变) ...
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