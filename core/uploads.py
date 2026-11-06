import os
import tempfile


def reset_upload_state(state):
    """上传组件发生变更时清除旧结果，包含同名替换和移除文件。"""
    for key in ('processing_started', 'processing_has_failed', 'stop_requested',
                'refinement_in_progress', 'refinement_stop_requested'):
        state[key] = False
    for key in ('current_notes', 'full_transcript', 'transcript_output_path',
                'current_assets_dir', 'current_output_dir', 'last_uploaded_filename'):
        state[key] = None


def save_uploaded_file(uploaded_file, upload_dir=os.path.join("temp", "uploads")):
    """为本次上传创建独立文件，避免复用或覆盖历史上传。"""
    os.makedirs(upload_dir, exist_ok=True)
    suffix = os.path.splitext(uploaded_file.name)[1]
    file_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=suffix, prefix="upload_", dir=upload_dir, delete=False
        ) as target:
            file_path = target.name
            target.write(uploaded_file.getbuffer())
        return file_path
    except BaseException:
        if file_path is not None:
            os.remove(file_path)
        raise
