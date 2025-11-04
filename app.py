# app.py
import streamlit as st
import os
from main import main_process_generator
# from config import DIFY_API_KEY # <-- Dify Key 不再需要
from config import ERNIE_CONFIG # <-- 仅为加载配置，确保 main 能访问

st.set_page_config(page_title="智能笔记 Agent", layout="wide")
st.title("👨‍💻 智能内容生成 Agent")
st.markdown("上传您的视频、音频或文本文档，即可自动生成结构化笔记、Q&A 或测验。")

# --- (修改) 更新提示信息 ---
st.info("💡 **提示**: 视频/音频文件将使用本地 Whisper 转录，笔记生成将调用 ERNIE API。")

with st.sidebar:
    st.header("⚙️ 参数配置")

    output_filename = st.text_input("请输入希望的笔记文件名 (无需后缀)", value="我的学习笔记")

    query_option = st.selectbox(
        "请选择生成内容类型:",
        ("Notes", "Q&A", "Quiz"),
        index=0,
        help="选择 'Notes' 生成结构化笔记, 'Q&A' 生成问答对, 'Quiz' 生成测验题。(目前仅 'Notes' 功能已接入 ERNIE)"
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

if uploaded_file is not None:
    if st.button("开始生成", use_container_width=True, type="primary"):

        st.markdown("---")
        st.subheader("处理进度")
        
        main_progress_bar = st.progress(0)
        main_progress_text = st.empty()
        sub_progress_bar = st.progress(0)
        sub_progress_text = st.empty()

        st.markdown("---")

        processing_headers = {
            "Notes": "正在生成笔记 (ERNIE 非流式)...",
            "Q&A": "正在进行 Q&A...",
            "Quiz": "正在生成测验..."
        }
        st.subheader(processing_headers.get(query_option, "正在处理..."))
        st.info(f"当前生成模式: **{query_option}**")
        
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

        # (注意: DIFY_API_KEY 参数现在是多余的，但 main_process_generator 仍需此参数)
        generator = main_process_generator(temp_file_path, "DUMMY_KEY", output_filename, query_option)
        
        for event_type, value, *rest in generator:
            text = rest[0] if rest else ""

            if event_type == "progress":
                main_progress_bar.progress(float(value))
                main_progress_text.info(text)
            elif event_type == "sub_progress":
                sub_progress_bar.progress(float(value))
                sub_progress_text.text(text)
            
            # --- (移除) 不再处理 'display_classification' 事件 ---
            # elif event_type == "display_classification":
            #     classification_display.success(f"✅ **笔记分类**: {value}")

            elif event_type == "llm_chunk":
                # 因为是非流式，value 将是完整的笔记内容
                full_llm_response += value
                llm_output_container.markdown(full_llm_response) # 直接显示完整内容
            
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