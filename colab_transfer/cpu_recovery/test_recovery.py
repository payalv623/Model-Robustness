"""End-to-end synthetic recovery test; does not claim real CPU inference validation."""
import io,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
SUPPORT=Path(__file__).resolve().parent
PROJECT=SUPPORT.parent/'project'
PHASE=r'''
import io,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import background_dataset as d
from color_dataset import ROOT,IDS,SPLITS,save_json,csv_bytes,atomic_bytes,sha,load_sources
from audit_background_dataset import audit
class Engine:
    config={'test':True}
    calls=0
    def segment(self,image,class_id,annotation=None):
        Engine.calls+=1
        mask=np.zeros((image.height,image.width),bool);mask[3:-3,3:-3]=True
        c={'predicted_iou':.97,'stability':.98,'area_fraction':float(mask.mean())}
        device='cuda:0' if sys.argv[1]=='original' else 'cpu'
        return mask,{'flags':['fixture_requires_review'],'detector_parameter_device':device,'sam_parameter_device':device,'sam_embedding_device':device,'instances':[{'mask_offset':0,'selected_candidate':0,'candidates':[c]}]},[mask]
d.ForegroundEngine=Engine
if sys.argv[1]=='original':
    rows=[]
    for i,c in enumerate(IDS):
        for si,s in enumerate(SPLITS):
            for n in range(10):
                ss='val.X' if s=='test' else 'train.X1'
                p=Path('archive')/ss/c/f'{s}_{n}.png'
                im=Image.fromarray(np.random.default_rng(i*100+si*10+n).integers(0,256,(20,24,3),dtype=np.uint8));b=io.BytesIO();im.save(b,format='PNG')
                atomic_bytes(ROOT/p,b.getvalue())
                rows.append({'path':str(p),'sha256':sha(b.getvalue()),'class_id':c,'class_index':i,'group':'A' if i<5 else 'B','split':s,'source_split':ss})
    b=csv_bytes(rows,tuple(rows[0]));atomic_bytes(ROOT/'project_metadata/source_splits.csv',b)
    save_json(ROOT/'project_metadata/split_config.json',{'split_manifest_sha256':sha(b),'per_class_counts':{c:{s:10 for s in SPLITS} for c in IDS}})
    save_json(ROOT/'project_metadata/background_annotations.json',{'images':{}})
    atomic_bytes(ROOT/'checkpoints/sam_vit_b_01ec64.pth',b'fixture checkpoint')
    d.run(full=True)
    out,report=d.paths(True)
    paths=sorted((out/'records').glob('*.json'))
    paths[0].unlink() # Incomplete item must be generated, not skipped.
    r=json.loads(paths[1].read_text());(out/(r['directory']+'/nature.png')).write_bytes(b'corrupt')
    assert Engine.calls==300
else:
    d.run(full=True)
    assert Engine.calls==2,Engine.calls
    out,report=d.paths(True)
    result=audit(out,report)
    assert result['status']=='numeric_audit_passed',result['issues'][:3]
    assert result['pixel_verified_pngs']==1200
    assert result['manifest_rows']==2400
    assert not result['training_ready'] and not (out/'manifests').exists()
    d.run(full=True);assert Engine.calls==2
    p=next((out/'imported_cuda_records').glob('*.json'));original=p.read_bytes();p.write_bytes(b'tampered')
    assert audit(out,report)['status']=='failed'
    p.write_bytes(original)
    from colab_checkpoint import Backup,restore
    import os
    os.environ['TEMPLATE2_BACKUP_DIR']=str(ROOT.parent/'fake_drive')
    Backup(ROOT,out).save(force=True)
    restored=ROOT.parent/'restored';restore(ROOT.parent/'fake_drive',restored)
    assert (restored/p.relative_to(ROOT)).read_bytes()==original
    assert (restored/out.relative_to(ROOT)/'imported_cuda_config.json').exists()
    print('PASS: 298 unchanged CUDA imports; 2 CPU repairs; resume; provenance tamper detection; 1200 pixel checks; 2400 manifest rows; checkpoint restore; training remains gated.')
'''
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);parent=root/'template2_project';parent.mkdir()
    for p in PROJECT.glob('*.py'):shutil.copyfile(p,parent/p.name)
    shutil.copytree(PROJECT/'project_metadata',parent/'project_metadata')
    (parent/'color_bias_dataset').mkdir()
    (parent/'test_phase.py').write_text(PHASE)
    subprocess.run([sys.executable,str(parent/'test_phase.py'),'original'],check=True,stdout=subprocess.DEVNULL)
    sys.path.insert(0,str(SUPPORT));from prepare_cpu_recovery import prepare
    target=root/'template2_cpu_recovery';prepare(parent,target,SUPPORT)
    with (SUPPORT/'test_recovery.log').open('w') as log:
        subprocess.run([sys.executable,str(target/'test_phase.py'),'cpu'],check=True,stdout=log,stderr=subprocess.STDOUT)
    print((SUPPORT/'test_recovery.log').read_text().splitlines()[-1])
