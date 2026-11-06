import os
import tempfile


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
