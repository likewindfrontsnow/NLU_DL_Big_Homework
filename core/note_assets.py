import os
import uuid


def create_assets_dir(output_dir):
    directory = os.path.join(output_dir, 'assets', uuid.uuid4().hex)
    os.makedirs(directory, exist_ok=False)
    return directory


def asset_relative_path(assets_dir, filename):
    return f"assets/{os.path.basename(assets_dir)}/{filename}"


def add_assets_to_zip(archive, assets_dir):
    """仅打包本次生成的图片，路径与笔记中的链接一致。"""
    for filename in os.listdir(assets_dir):
        full_path = os.path.join(assets_dir, filename)
        if os.path.isfile(full_path):
            archive.write(full_path, asset_relative_path(assets_dir, filename))
