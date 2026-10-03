import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from artifact_bundle import INDEX, pack, restore, sha


class ArtifactBundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve() / 'source'
        self.root.mkdir()
        self.exports = self.root.parent / 'exports'
        self.manifest = self.root.parent / 'manifest.json'
        self.destination = self.root.parent / 'restored'

    def bundle(self, path, size=1000, part=None):
        return pack(self.root, {'id': 'test', 'local_path': path}, self.exports, self.manifest, size, part)

    def read_manifest(self):
        return json.loads(self.manifest.read_text())

    def test_zip_roundtrip_reuses_parts_and_refuses_changed_destination(self):
        folder = self.root / 'data'; folder.mkdir()
        for i in range(3): (folder / f'{i}.bin').write_bytes(bytes([i]) * 600)
        result = self.bundle('data')
        self.assertEqual(len(result['parts']), 3)
        self.assertEqual(self.bundle('data'), result)
        restore(self.read_manifest(), self.exports, self.destination)
        for p in folder.iterdir(): self.assertEqual(p.read_bytes(), (self.destination / 'data' / p.name).read_bytes())
        (self.destination / 'data/0.bin').write_bytes(b'edited')
        with self.assertRaisesRegex(ValueError, 'Refusing to replace'):
            restore(self.read_manifest(), self.exports, self.destination)
        self.assertEqual((self.destination / 'data/0.bin').read_bytes(), b'edited')

    def test_split_large_file_requires_all_parts_and_reassembles(self):
        source = self.root / 'large.zip'; source.write_bytes(bytes(range(256)) * 10)
        self.bundle('large.zip', part=1)
        with self.assertRaisesRegex(ValueError, 'Incomplete group'):
            restore(self.read_manifest(), self.exports, self.destination)
        self.bundle('large.zip')
        restore(self.read_manifest(), self.exports, self.destination)
        self.assertEqual(source.read_bytes(), (self.destination / 'large.zip').read_bytes())
        restore(self.read_manifest(), self.exports, self.destination)

    def test_corrupted_download_fails_before_writing(self):
        (self.root / 'file').write_bytes(b'original')
        group = self.bundle('file')
        path = self.exports / group['parts'][0]['file']
        path.write_bytes(path.read_bytes() + b'tampered')
        with self.assertRaisesRegex(ValueError, 'corrupted'):
            restore(self.read_manifest(), self.exports, self.destination)
        self.assertFalse(self.destination.exists())

    def test_traversal_and_symlink_destinations_rejected(self):
        (self.root / 'file').write_bytes(b'original')
        self.bundle('file')
        manifest = self.read_manifest(); part = manifest['groups']['test']['parts'][0]
        path = self.exports / part['file']
        with zipfile.ZipFile(path, 'w') as archive:
            archive.writestr('../escape', b'bad')
            archive.writestr(INDEX, json.dumps([{'path': '../escape', 'bytes': 3, 'sha256': 'invalid'}]))
        part.update(bytes=path.stat().st_size, sha256=sha(path))
        with self.assertRaisesRegex(ValueError, 'Unsafe artifact path'):
            restore(manifest, self.exports, self.destination)
        self.destination.mkdir()
        (self.destination / 'linked').symlink_to(self.root, target_is_directory=True)
        from artifact_bundle import safe
        with self.assertRaisesRegex(ValueError, 'Symlink destination'):
            safe(self.destination, 'linked/file')
