"""Synthetic end-to-end test of mixed-backend merge and corruption detection."""
import ast,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
SUPPORT=Path(__file__).resolve().parent;BUNDLE=SUPPORT.parent/'project';MAC=SUPPORT.parent/'mac_remainder'
source=ast.parse((SUPPORT.parent/'cpu_recovery/test_recovery.py').read_text())
phase=next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PHASE' for t in n.targets))
phase=phase.replace('    paths[0].unlink()', '    # paths[0].unlink()').replace("    r=json.loads(paths[1].read_text());(out/(r['directory']+'/nature.png')).write_bytes(b'corrupt')",'    pass')
with tempfile.TemporaryDirectory() as temp:
    d=Path(temp);cuda=d/'cuda';cuda.mkdir()
    for p in BUNDLE.glob('*.py'):shutil.copyfile(p,cuda/p.name)
    shutil.copytree(BUNDLE/'project_metadata',cuda/'project_metadata');(cuda/'color_bias_dataset').mkdir()
    (cuda/'fixture.py').write_text(phase)
    subprocess.run([sys.executable,str(cuda/'fixture.py'),'original'],check=True,stdout=subprocess.DEVNULL)
    mac=d/'mac';mac.mkdir()
    for p in MAC.glob('*.py'):shutil.copyfile(p,mac/p.name)
    shutil.copytree(cuda/'project_metadata',mac/'project_metadata')
    out=cuda/'background_bias_dataset/production_v3_cuda';mout=mac/'background_bias_dataset/production_v3_mac_remainder'
    mout.mkdir(parents=True)
    import hashlib
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    cfg=json.loads((out/'config.json').read_text());cfg['full']=False
    cfg['policy']={**cfg['policy'],'detector_device':'cpu','sam_device':'mps'}
    cfg['foreground_code_sha256']=h(mac/'background_foreground.py');cfg['generator_code_sha256']=h(mac/'background_dataset.py')
    (mout/'config.json').write_text(json.dumps(cfg))
    # Create two valid MPS original records, matching fixture row types.
    sys.path.insert(0,str(cuda));from color_dataset import load_sources
    rows,_,_=load_sources(cuda)
    for row in rows[:2]:
        key=hashlib.sha256(row['path'].encode()).hexdigest()[:20];r=json.loads((out/'records'/f'{key}.json').read_text())
        r['segmentation'].update(detector_parameter_device='cpu',sam_parameter_device='mps:0',sam_embedding_device='mps:0')
        r['identity']=hashlib.sha256(json.dumps({'config':cfg,'source':row,'annotation':None},sort_keys=True).encode()).hexdigest()
        for rel in r['file_hashes']:
            (mout/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(out/rel,mout/rel)
        (mout/'records').mkdir(exist_ok=True);(mout/'records'/f'{key}.json').write_text(json.dumps(r))
    target=d/'merged';log=SUPPORT/'test_merge.log'
    env={k:v for k,v in os.environ.items() if k!='TEMPLATE2_BACKUP_DIR'}
    with log.open('w') as f:subprocess.run([sys.executable,str(SUPPORT/'merge_results.py'),str(cuda),str(mac),str(target)],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
    report=target/'reports/background_bias_current/production_v3_merged_cuda_mps';result=json.loads((report/'audit.json').read_text())
    assert result['source_images_verified']==300 and result['pixel_verified_pngs']==1200
    assert result['manifest_rows']==2400 and result['backend_source_counts']=={'cuda':298,'mps':2}
    assert not result['training_ready'] and result['quality_review_unresolved']==300
    provenance=next((target/'background_bias_dataset/production_v3_merged_cuda_mps/provenance/mps/records').glob('*.json'));provenance.write_text('{}')
    p=subprocess.run([sys.executable,str(target/'audit_background_dataset.py'),'--full'],stdout=subprocess.DEVNULL,env=env)
    assert p.returncode!=0,'Corrupt original provenance passed'
    print('PASS: 298 CUDA + 2 MPS sources; 1200 pixel checks; 2400 manifest rows; review gates retained; altered provenance rejected.')
