# app.py
import streamlit as st
import os
import time
from main import main_process_generator
from config import LLM_CONFIG 
from llm_api import refine_llm_generation 
from dotenv import set_key

st.set_page_config(page_title="智能笔记 Agent", layout="wide")
st.title("👨‍💻 智能内容生成 Agent")

def _save_key(key):
    env_path = ".env"
    set_key(env_path, "LLM_API_KEY", key)
    set_key(env_path, "DASHSCOPE_API_KEY", key)
    LLM_CONFIG["api_key"] = key
    if not LLM_CONFIG.get("provider_name"):
        LLM_CONFIG["provider_name"] = "DashCope(Qwen)"
        set_key(env_path, "LLM_PROVIDER_NAME", "DashCope(Qwen)")


if not LLM_CONFIG.get("api_key"):
    st.warning("🔑 未检测到 API Key")
    st.markdown("""
    请输入您的 DashScope (Qwen) API Key。
    
    - 您只需要输入一次，Key 将被安全地保存到项目根目录的 `.env` 文件中。
    - 您的 Key 仅用于您本地的程序调用，不会上传到别处。
    - 您可以前往 [阿里云 DashScope 控制台](https://dashscope.console.aliyun.com/apiKey) 获取您的 Key。
    """)
    
    new_key = st.text_input(
        "请输入您的 Qwen API Key (sk-xxxxxxxx)", 
        type="password", 
        key="api_key_input_main"
    )
    
    if st.button("保存并开始", use_container_width=True, type="primary"):
        if new_key and new_key.startswith("sk-"):
            _save_key(new_key)
            st.success("API Key 已保存！正在重启应用...")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Key 格式不正确。它应该以 `sk-` 开头。")

else:
    st.markdown("上传您的视频、音频或文本文档，即可自动生成结构化笔记。")

    if "processing_started" not in st.session_state:
        st.session_state.processing_started = False
    if "current_notes" not in st.session_state:
        st.session_state.current_notes = None
    if "full_transcript" not in st.session_state:
        st.session_state.full_transcript = None
    if "output_filename" not in st.session_state:
        st.session_state.output_filename = "我的学习笔记"
    if "last_uploaded_filename" not in st.session_state:
        st.session_state.last_uploaded_filename = None
    if "processing_has_failed" not in st.session_state:
        st.session_state.processing_has_failed = False
    if "api_key_invalid" not in st.session_state:
        st.session_state.api_key_invalid = False
    if "asr_context" not in st.session_state:
        st.session_state.asr_context = ""
    if "note_type" not in st.session_state:
        st.session_state.note_type = "STEM"
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False
    if "refinement_in_progress" not in st.session_state:
        st.session_state.refinement_in_progress = False
    if "refinement_stop_requested" not in st.session_state:
        st.session_state.refinement_stop_requested = False

    if "preset_feedback" not in st.session_state:
        st.session_state.preset_feedback = "(请选择一个快捷指令)"
    if "custom_feedback" not in st.session_state:
        st.session_state.custom_feedback = ""

    is_busy = st.session_state.processing_started or st.session_state.refinement_in_progress

    provider_name = LLM_CONFIG.get('provider_name', 'LLM')
    st.info(f"💡 **提示**: 视频/音频文件将使用所选转录服务，笔记生成将调用 **{provider_name}** API。")

    with st.sidebar:
        st.header("⚙️ 参数配置")

        st.session_state.output_filename = st.text_input(
            "请输入希望的笔记文件名 (无需后缀)", 
            value=st.session_state.output_filename,
            disabled=is_busy
        )
        
        note_type_option = st.radio(
            "请选择笔记类型:",
            ("STEM", "HASS"),
            index=0, 
            key="note_type", 
            horizontal=True,
            help="STEM (理工科) 适用于数学/代码/科学。HASS (人文社科) 适用于历史/文学/社会学。",
            disabled=is_busy
        )

        st.markdown("---")
        st.subheader("语音转录 (ASR) 配置")
        transcription_provider = st.radio(
            "请选择语音转录服务:",
            ("Local Whisper", "Qwen API"),
            index=0,
            key="transcription_provider",
            help="""
            - **Local Whisper**: 在您本地电脑上运行，速度取决于您的电脑配置，首次加载较慢。
            - **Qwen API**: 调用阿里云 Qwen API，相比Local Whisper速度更快，精度更高。
            """,
            disabled=is_busy
        )
        
        whisper_model_size = "tiny" 
        if transcription_provider == "Local Whisper":
            whisper_model_size = st.selectbox(
                "请选择 Whisper 模型:",
                ("tiny", "base", "small", "medium", "large"),
                index=0,
                help="模型越大，转录越准确，但速度越慢。'tiny' 最快，'large' 最准。首次使用模型时，程序会先下载模型文件（可能需要几分钟）。",
                disabled=is_busy
            )
            if st.session_state.asr_context != "":
                st.session_state.asr_context = "" 
        else:
            st.info("Qwen API 将使用 qwen3-asr-flash 模型。")
        
        if transcription_provider == "Qwen API":
            st.markdown("---")
            st.subheader("ASR 上下文增强 (Qwen)")
            st.session_state.asr_context = st.text_area(
                "输入热词 (用于提升 ASR 准确率)",
                value=st.session_state.asr_context,
                placeholder="例如: Bulge Bracket, Boutique, 投行...",
                help="在此处输入希望 Qwen API 优先识别的专业词汇、人名或地名，用逗号或段落分隔均可。",
                disabled=is_busy
            )

        st.markdown("---")
        stream_output = st.toggle(
            "启用笔记流式输出", 
            value=True, 
            help="启用后，笔记内容将实时逐字显示。禁用则会在所有内容生成后一次性显示。",
            disabled=is_busy
        )

        st.markdown("---")
        keep_temp_files = st.checkbox(
            "保留语音转文字稿", 
            value=False, 
            help="勾选后将保留语音转文字生成的 .txt 文字稿，上传的原始文件总会被自动删除。",
            disabled=is_busy
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
        type=all_exts,
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
        st.subheader(f"正在生成笔记 ({provider_name} {stream_status})...")
        
        if transcription_provider == "Local Whisper":
            asr_config_text = f"转录服务: **Local Whisper** (模型: **{whisper_model_size}**)"
        else:
            asr_config_text = "转录服务: **Qwen API** (模型: **qwen3-asr-flash**)"
            if st.session_state.asr_context:
                asr_config_text += f" | **上下文:** *{st.session_state.asr_context[:30]}...*"
        
        st.info(f"{asr_config_text} | 笔记类型: **{st.session_state.note_type}**")
        
        llm_output_container = st.empty()
        full_llm_response = ""
        
        final_result_path = None
        
        temp_dir = "temp_uploads"
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
            st.session_state.asr_context
        )
        
        for event_type, value, *rest in generator:
            
            if st.session_state.get("stop_requested", False):
                st.session_state.stop_requested = False
                st.session_state.processing_started = False
                st.warning("处理已由用户手动停止。")
                main_progress_text.warning("处理已停止。")
                stop_button_placeholder.empty()
                st.rerun()
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
            
            elif event_type == "persistent_error":
                st.error(f"处理失败: {text}")
                main_progress_text.error("一个关键步骤在多次重试后仍然失败，已停止处理。")
                llm_output_container.error(f"**错误详情:**\n\n{text}")
                
                if "401" in text or "密钥无效" in text or "Unauthorized" in text:
                    st.session_state.api_key_invalid = True
                
                st.session_state.processing_has_failed = True 
                st.session_state.processing_started = False
                stop_button_placeholder.empty()
                st.rerun() 
                break
            
            elif event_type == "error":
                st.error(text)
                llm_output_container.error(text)
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
                final_result_path = value
                
                st.session_state.current_notes = full_llm_response
                st.session_state.processing_started = False
                stop_button_placeholder.empty()
                
                if not keep_temp_files:
                    transcript_path = "source_transcript.txt"
                    try:
                        if os.path.exists(transcript_path):
                            os.remove(transcript_path)
                    except OSError as e:
                        st.warning(f"无法自动删除文字稿文件 '{transcript_path}': {e}")
                else:
                    st.info("已根据您的设置，保留了语音转文字稿。")
                
                try:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                except OSError as e:
                    st.warning(f"无法自动删除临时上传文件 '{temp_file_path}': {e}")
                
                st.rerun() 
                break


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
                    save_path = f"{st.session_state.output_filename}_refined.md"
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