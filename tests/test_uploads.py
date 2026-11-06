import io
import tempfile
import unittest
from pathlib import Path

from core.uploads import save_uploaded_file


class UploadTests(unittest.TestCase):
    def upload(self, content):
        uploaded = io.BytesIO(content)
        uploaded.name = "lecture.mp4"
        return uploaded

    def test_existing_video_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            stale = Path(directory) / "lecture.mp4"
            stale.write_bytes(b"old video")
            uploaded = self.upload(b"current video")
            uploaded.read()  # 保存完整内容，不受上传流当前位置影响。
            saved = Path(save_uploaded_file(uploaded, directory))
            self.assertEqual(saved.read_bytes(), b"current video")
            self.assertEqual(saved.suffix, ".mp4")
            self.assertEqual(stale.read_bytes(), b"old video")

    def test_same_name_uploads_have_independent_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(save_uploaded_file(self.upload(b"first"), directory))
            second = Path(save_uploaded_file(self.upload(b"second"), directory))
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), b"first")
            self.assertEqual(second.read_bytes(), b"second")

    def test_failed_upload_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            uploaded = self.upload(b"video")
            uploaded.close()
            with self.assertRaises(ValueError):
                save_uploaded_file(uploaded, directory)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
