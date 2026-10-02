"""Create a separate CPU continuation with explicit, verifiable CUDA import provenance."""
from pathlib import Path
import hashlib,json,shutil


def replace_once(text,old,new):
    if old not in text:raise ValueError('Unexpected source version: '+old[:70])
    return text.replace(old,new)


def prepare(parent, target, support):
    parent,target,support=map(Path,(parent,target,support))
    target.mkdir(parents=True,exist_ok=True)
    for p in parent.glob('*.py'):shutil.copyfile(p,target/p.name)
    for p in parent.glob('requirements*.txt'):shutil.copyfile(p,target/p.name)
    shutil.copytree(parent/'project_metadata',target/'project_metadata',dirs_exist_ok=True)
    shutil.copytree(parent/'color_bias_dataset',target/'color_bias_dataset',dirs_exist_ok=True)
    for name in ('archive','checkpoints'):
        if not (target/name).exists():(target/name).symlink_to(parent/name,target_is_directory=True)
    shutil.copyfile(support/'recovery_provenance.py',target/'recovery_provenance.py')
    original=parent/'background_bias_dataset/production_v3_cuda/config.json'
    parent_hash=hashlib.sha256(original.read_bytes()).hexdigest()
    p=target/'background_foreground.py';s=p.read_text()
    s=s.replace("'version': 3, 'detector_device': 'cuda', 'sam_device': 'cuda'", "'version': '3_cpu_recovery', 'detector_device': 'cpu', 'sam_device': 'cpu',\n    'allowed_record_backends': ['cuda_imported_with_provenance','cpu'],\n    'imported_cuda_config_sha256': "+repr(parent_hash))
    s=s.replace("select_device('cuda')", "select_device('cpu')")
    s=s.replace("torch.set_num_threads(4)", "torch.set_num_threads(min(4, os.cpu_count() or 2))")
    s=s.replace("torch.cuda.get_device_name(0)", "'CPU continuation'")
    s=s.replace(".to('cuda')", ".to('cpu')").replace("'cuda:0'", "'cpu'").replace('not on CUDA','not on CPU')
    s=s.replace('torch.cuda.synchronize()', 'pass  # CPU operations are synchronous')
    s=s.replace("'cuda_allocated_bytes':torch.cuda.memory_allocated()", "'cpu_execution':True")
    p.write_text(s)
    p=target/'background_dataset.py';s=p.read_text().replace(" + '_cuda'",'')
    s=s.replace("        save_json(cfg,config)", "        save_json(cfg,config)\n        from recovery_provenance import import_cuda\n        import_cuda(output,config,rows,annotations)")
    s=s.replace('Loading class detector and SAM on CUDA (no CPU fallback)','Loading CPU continuation for unfinished images only')
    s=s.replace("'sam_device':'cuda'", "'sam_device':'mixed_cuda_cpu'").replace("sam_device='cuda'", "sam_device='mixed_cuda_cpu'")
    p.write_text(s)
    p=target/'audit_background_dataset.py';s=p.read_text()
    old="            if info.get('detector_parameter_device')!='cuda:0' or info['sam_parameter_device']!='cuda:0' or info['sam_embedding_device']!='cuda:0':\n                raise ValueError('SAM device is not CUDA')"
    s=replace_once(s,old,"            from recovery_provenance import verify_backend\n            verify_backend(record,row,config,annotations.get(row['path']),output)")
    s=s.replace("'sam_device':'cuda'", "'sam_device':'mixed_cuda_cpu'");p.write_text(s)
    p=target/'colab_checkpoint.py';s=p.read_text().replace('        self.saved = {}','        self.saved = {}\n        self.calls = 0')
    s=s.replace('        changed=[]', '        self.calls += 1\n        if not force and self.calls % 10:\n            return\n        changed=[]')
    s=s.replace('len(changed)<100', 'len(changed)<10')
    s=s.replace("destination.glob('*_cuda')", "destination.glob('*cpu_recovery')")
    p.write_text(s)
    # CPU continuation uses full source selection, never a new sampling or correlation scheme.
    runner='''import json,time\nfrom background_dataset import run,paths\nfrom audit_background_dataset import audit\nfrom color_dataset import ROOT,save_json\nfrom colab_checkpoint import Backup\n\nif __name__=='__main__':\n    out,report=paths(True)\n    try:\n        run(full=True)\n        result=audit(out,report)\n        print(json.dumps({k:v for k,v in result.items() if k not in ('issues','unresolved_sources')},indent=2),flush=True)\n        if result['issues']:raise RuntimeError('Full numeric audit has unresolved issues; see audit.json')\n    finally:\n        Backup(ROOT,out).save(force=True)\n'''
    (target/'run_cpu_recovery.py').write_text(runner)
    (target/'background_progress_report.py').write_text('''import html\nfrom color_dataset import atomic_bytes\ndef render(report,status):\n    atomic_bytes(report/'index.html',('<h1>Template 2 CPU continuation</h1><p>Verified CUDA outputs are retained unchanged; unfinished sources use CPU float32. Numerical checks and visual review remain required.</p><pre>'+html.escape(str(status))+'</pre>').encode())\n''')
    print('Prepared CPU continuation:',target)

if __name__=='__main__':
    import sys
    prepare(*sys.argv[1:])
