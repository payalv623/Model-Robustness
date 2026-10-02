import json,os,time,zipfile
from background_dataset import run,paths
from audit_background_dataset import audit
from color_dataset import ROOT,save_json,sha

if __name__=='__main__':
    out,report=paths(True)
    save_json(report/'worker.json',{'pid':os.getpid(),'sources':1102,'device':'MPS SAM, CPU detector','full_dataset':False})
    run(full=True)
    result=audit(out,report)
    save_json(report/'audit.json',result)
    # Failed sources remain in the archive/report; merging may use only generated records.
    destination=ROOT.parent/'template2_mac_remainder.zip'
    with zipfile.ZipFile(destination.with_suffix('.zip.partial'),'w',zipfile.ZIP_STORED) as z:
        for folder in (out,report,ROOT/'project_metadata'):
            for p in sorted(folder.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(ROOT))
        for p in sorted(ROOT.glob('*.py')):z.write(p,p.name)
    destination.with_suffix('.zip.partial').replace(destination)
    save_json(ROOT.parent/'template2_mac_remainder_manifest.json',{'archive':destination.name,'sha256':sha(destination.read_bytes()),'numeric_audit_status':result['status'],'training_ready':False})
    print('Remainder archive ready:',destination,flush=True)
