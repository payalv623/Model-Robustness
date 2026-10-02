import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from download_models import fetch


class ModelDownloadTests(unittest.TestCase):
    def test_verified_download_and_existing_file_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'source', root / 'models/model.pth'
            source.write_bytes(b'checkpoint bytes')
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            fetch(source.as_uri(), target, expected)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            with patch('download_models.urllib.request.urlretrieve', side_effect=AssertionError('unexpected download')):
                fetch(source.as_uri(), target, expected)

    def test_bad_download_preserves_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'source', root / 'model.pth'
            source.write_bytes(b'invalid download')
            target.write_bytes(b'existing checkpoint')
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                fetch(source.as_uri(), target, hashlib.sha256(b'expected').hexdigest())
            self.assertEqual(target.read_bytes(), b'existing checkpoint')
