"""Prepare a separate MPS worker for the tail absent from the confirmed 12,400-record cloud checkpoint."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2]
TARGET=ROOT/'colab_transfer/mac_remainder'
TARGET.mkdir(exist_ok=True)
for name in ('color_dataset.py','background_dataset.py','background_foreground.py','background_grounded_pilot.py','background_pilot.py','background_scenes.py','background_progress_report.py','audit_background_dataset.py'):
    shutil.copyfile(ROOT/name,TARGET/name)
shutil.copytree(ROOT/'project_metadata',TARGET/'project_metadata',dirs_exist_ok=True)
for name in ('archive','checkpoints'):
    if not (TARGET/name).exists():(TARGET/name).symlink_to(ROOT/name,target_is_directory=True)
assert hashlib.sha256((TARGET/'project_metadata/source_splits.csv').read_bytes()).hexdigest()=='61591c06294ed7659446e90cd970088cf8a0cff834feed4e8816bc842454f6e8'
import sys
sys.path.insert(0,str(ROOT))
from color_dataset import load_sources
rows,_,_=load_sources(ROOT)
missing=rows[12400:]
failed_names={'n01484850_20518.JPEG','n01775062_8018.JPEG'}
failures=[r for r in rows[:12400] if Path(r['path']).name in failed_names]
assert len(missing)==1100 and len(failures)==2
selected_paths={r['path'] for r in missing+failures}
selected=[r for r in rows if r['path'] in selected_paths]
(TARGET/'project_metadata/recovery_sources.json').write_text(json.dumps({'reason':'Frozen traversal suffix after confirmed Drive checkpoint of 12400 processed records, plus two failed sources observed in CPU retry','source_count':len(selected),'source_paths':[r['path'] for r in selected]},indent=2)+'\n')
p=TARGET/'background_dataset.py';s=p.read_text()
s=s.replace("suffix = ('production_v' if full else 'refined_pilot_v') + str(POLICY['version'])", "suffix = 'production_v3_mac_remainder'")
old="    if not full:\n        rows=[r for c in IDS for r in [x for x in rows if x['class_id']==c and x['split']=='train'][:pilot_per_class]]"
assert old in s
s=s.replace(old,"    selection=json.loads((ROOT/'project_metadata/recovery_sources.json').read_text())['source_paths']\n    row_by_path={r['path']:r for r in rows}\n    rows=[row_by_path[p] for p in selection]")
s=s.replace("'full':full,", "'full':False, 'partial_recovery':True,")
s=s.replace("summary=export_manifests(output,report,rows) if full else", "summary=export_manifests(output,report,rows) if False else")
s=s.replace("'pilot_generated_pending_review'", "'partial_recovery_generated_pending_review'")
s=s.replace("'full_run':full", "'full_run':False")
p.write_text(s)
(TARGET/'run_mac_remainder.py').write_text('''import json,os,time,zipfile\nfrom background_dataset import run,paths\nfrom audit_background_dataset import audit\nfrom color_dataset import ROOT,save_json,sha\n\nif __name__=='__main__':\n    out,report=paths(True)\n    save_json(report/'worker.json',{'pid':os.getpid(),'sources':1102,'device':'MPS SAM, CPU detector','full_dataset':False})\n    run(full=True)\n    result=audit(out,report)\n    save_json(report/'audit.json',result)\n    # Failed sources remain in the archive/report; merging may use only generated records.\n    destination=ROOT.parent/'template2_mac_remainder.zip'\n    with zipfile.ZipFile(destination.with_suffix('.zip.partial'),'w',zipfile.ZIP_STORED) as z:\n        for folder in (out,report,ROOT/'project_metadata'):\n            for p in sorted(folder.rglob('*')):\n                if p.is_file():z.write(p,p.relative_to(ROOT))\n        for p in sorted(ROOT.glob('*.py')):z.write(p,p.name)\n    destination.with_suffix('.zip.partial').replace(destination)\n    save_json(ROOT.parent/'template2_mac_remainder_manifest.json',{'archive':destination.name,'sha256':sha(destination.read_bytes()),'numeric_audit_status':result['status'],'training_ready':False})\n    print('Remainder archive ready:',destination,flush=True)\n''')
print('Prepared',len(selected),'sources in',TARGET)
