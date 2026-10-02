import json, os, tempfile, unittest, zipfile
from pathlib import Path
from unittest.mock import patch
from colab_checkpoint import Backup, restore, extract, digest

class CheckpointTests(unittest.TestCase):
    def test_save_restore_repair_and_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td); root=d/'source'; out=root/'background_bias_dataset/production_v3_cuda'
            (out/'records').mkdir(parents=True);(out/'images').mkdir()
            f=out/'images/mask.png';f.write_bytes(b'first')
            record=out/'records/a.json';record.write_text(json.dumps({'file_hashes':{'images/mask.png':digest(f)}}))
            with patch.dict(os.environ,{'TEMPLATE2_BACKUP_DIR':str(d/'drive')}):
                b=Backup(root,out);b.save();self.assertFalse(list((d/'drive').rglob('*.zip')))
                b.save(force=True)
                target=d/'restore';restore(d/'drive',target)
                self.assertEqual((target/f.relative_to(root)).read_bytes(),b'first')
                f.write_bytes(b'repaired');record.write_text(json.dumps({'file_hashes':{'images/mask.png':digest(f)}}))
                Backup(root,out).save(force=True);restore(d/'drive',target)
                self.assertEqual((target/f.relative_to(root)).read_bytes(),b'repaired')
                z=next((d/'drive').rglob('*.zip'));z.write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError,'hash mismatch'):restore(d/'drive',target)
    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);z=d/'evil.zip'
            with zipfile.ZipFile(z,'w') as f:f.writestr('../escape','bad')
            with self.assertRaises(ValueError):extract(z,d/'target')
            self.assertFalse((d/'escape').exists())

if __name__=='__main__':unittest.main()
