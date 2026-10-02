"""Merge completed CUDA + MPS candidates with immutable original provenance; never infer new masks."""
import hashlib,json,os,shutil,subprocess,sys
from pathlib import Path

def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def jwrite(p,data):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')

def prepare(cuda,mac,target):
    support=Path(__file__).resolve().parent
    cuda,mac,target=map(lambda p:Path(p).resolve(),(cuda,mac,target))
    source_outputs={'cuda':cuda/'background_bias_dataset/production_v3_cuda','mps':mac/'background_bias_dataset/production_v3_mac_remainder'}
    configs={k:json.loads((v/'config.json').read_text()) for k,v in source_outputs.items()}
    for provider,root in [('cuda',cuda),('mps',mac)]:
        c=configs[provider]
        for field,relative in [('foreground_code_sha256','background_foreground.py'),('generator_code_sha256','background_dataset.py'),('source_split_sha256','project_metadata/source_splits.csv'),('detector_provenance_sha256','project_metadata/grounding_dino_checkpoint.json')]:
            if digest(root/relative)!=c[field]:raise ValueError('Input code/metadata hash mismatch: '+provider+'/'+relative)
    semantic=lambda c:{k:v for k,v in c['policy'].items() if k not in ('version','detector_device','sam_device')}
    if semantic(configs['cuda'])!=semantic(configs['mps']):raise ValueError('Segmentation policies differ')
    for k in ('source_split_sha256','sam_checkpoint_sha256','detector_provenance_sha256','correlation','seed','variants'):
        if configs['cuda'][k]!=configs['mps'][k]:raise ValueError('Input configuration differs: '+k)
    for name in ('background_scenes.py','color_dataset.py'):
        if digest(cuda/name)!=digest(mac/name):raise ValueError('Rendering/assignment code differs: '+name)
    target.mkdir(parents=True,exist_ok=True)
    for p in cuda.glob('*.py'):shutil.copyfile(p,target/p.name)
    shutil.copytree(cuda/'project_metadata',target/'project_metadata',dirs_exist_ok=True)
    shutil.copytree(cuda/'color_bias_dataset',target/'color_bias_dataset',dirs_exist_ok=True)
    for name in ('archive','checkpoints'):
        if not (target/name).exists():(target/name).symlink_to(cuda/name,target_is_directory=True)
    for name in ('merge_provenance.py','merge_results.py'):shutil.copyfile(support/name,target/name)
    s=(target/'background_dataset.py').read_text()
    s=s.replace("suffix = ('production_v' if full else 'refined_pilot_v') + str(POLICY['version']) + '_cuda'", "suffix = 'production_v3_merged_cuda_mps'")
    (target/'background_dataset.py').write_text(s)
    s=(target/'audit_background_dataset.py').read_text()
    old="            if info.get('detector_parameter_device')!='cuda:0' or info['sam_parameter_device']!='cuda:0' or info['sam_embedding_device']!='cuda:0':\n                raise ValueError('SAM device is not CUDA')"
    if old not in s:raise ValueError('Unexpected audit source')
    s=s.replace(old,"            from merge_provenance import verify_backend\n            verify_backend(record,row,config,annotations.get(row['path']),output)")
    s=s.replace("'sam_device':'cuda'","'sam_device':'per_record_cuda_or_mps'")
    (target/'audit_background_dataset.py').write_text(s)
    command=[sys.executable,str(target/'merge_results.py'),'--execute',str(cuda),str(mac)]
    subprocess.run(command,cwd=target,check=True)

def execute(cuda,mac):
    from color_dataset import ROOT,load_sources,save_json,sha
    from background_dataset import paths,key_for,fingerprint,cached_record,export_manifests
    from audit_background_dataset import audit
    from merge_provenance import verify_backend,safe
    roots={'cuda':Path(cuda),'mps':Path(mac)}
    outputs={'cuda':roots['cuda']/'background_bias_dataset/production_v3_cuda','mps':roots['mps']/'background_bias_dataset/production_v3_mac_remainder'}
    configs={k:json.loads((v/'config.json').read_text()) for k,v in outputs.items()}
    ann_files={k:roots[k]/'project_metadata/background_annotations.json' for k in roots}
    annotations={k:json.loads(p.read_text()).get('images',{}) for k,p in ann_files.items()}
    rows,_,_=load_sources(ROOT);out,report=paths(True);out.mkdir(parents=True,exist_ok=True)
    evidence={}
    for provider in roots:
        cfg=f'provenance/{provider}/config.json';ann=f'provenance/{provider}/annotations.json'
        for src,rel in [(outputs[provider]/'config.json',cfg),(ann_files[provider],ann)]:
            p=out/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,p)
        evidence[provider]={'config':cfg,'config_sha256':digest(out/cfg),'annotations':ann,'annotations_sha256':digest(out/ann)}
        evidence[provider]['support_files']={}
        engine=outputs[provider]/'engine_config.json'
        if engine.exists():
            rel=f'provenance/{provider}/engine_config.json';shutil.copyfile(engine,out/rel);evidence[provider]['support_files'][rel]=digest(out/rel)
        for name in ('background_dataset.py','background_foreground.py','background_scenes.py','color_dataset.py','repair_missing.py','run_mac_remainder.py'):
            if (roots[provider]/name).exists():
                rel=f'provenance/{provider}/{name}';shutil.copyfile(roots[provider]/name,out/rel);evidence[provider]['support_files'][rel]=digest(out/rel)
    cfg={**configs['cuda'],'policy':{**configs['cuda']['policy'],'version':'3_merged_cuda_mps','detector_device':'per_record','sam_device':'per_record'},'full':True,'merge_inputs':evidence,'merge_code_sha256':digest(Path(__file__)),'provenance_verifier_sha256':digest(ROOT/'merge_provenance.py')}
    if cfg['source_selection_sha256']!=sha(json.dumps(rows,sort_keys=True).encode()):raise ValueError('CUDA full selection mismatch')
    if (out/'config.json').exists() and json.loads((out/'config.json').read_text())!=cfg:raise ValueError('Existing merge has different configuration')
    save_json(out/'config.json',cfg)
    merged_annotations={};counts={'cuda':0,'mps':0};chosen={}
    for number,row in enumerate(rows,1):
        key=key_for(row);provider='mps' if (outputs['mps']/'records'/f'{key}.json').exists() else 'cuda'
        old=cached_record(outputs[provider],row,fingerprint(configs[provider],row,annotations[provider].get(row['path'])))
        if not old:raise ValueError('Missing, stale or corrupted chosen source: '+row['path'])
        annotation=annotations[provider].get(row['path'])
        if annotation:merged_annotations[row['path']]=annotation
        data=(outputs[provider]/'records'/f'{key}.json').read_bytes()
        proof_path=f'provenance/{provider}/records/{key}.json'
        proof=out/proof_path;proof.parent.mkdir(parents=True,exist_ok=True);proof.write_bytes(data)
        for relative,h in old['file_hashes'].items():
            source=safe(outputs[provider],relative);dest=safe(out,relative);dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():
                if digest(dest)!=h:raise ValueError('Existing merged file is corrupt: '+relative)
            else:
                try:os.link(source,dest)
                except OSError:shutil.copyfile(source,dest)
        record={**old,'identity':fingerprint(cfg,row,annotation),'file_hashes':dict(old['file_hashes']),'merge_provenance':{'provider':provider,'record':proof_path,'record_sha256':sha(data)}}
        record['file_hashes'][proof_path]=sha(data)
        for kind in ('config','annotations'):
            relative=evidence[provider][kind];record['file_hashes'][relative]=evidence[provider][kind+'_sha256']
        record['file_hashes'].update(evidence[provider]['support_files'])
        verify_backend(record,row,cfg,annotation,out)
        save_json(out/'records'/f'{key}.json',record);counts[provider]+=1;chosen[row['path']]=record
        if number%1000==0:print(f'Merged and verified {number}/{len(rows)} source records',flush=True)
    save_json(ROOT/'project_metadata/background_annotations.json',{'schema':1,'images':merged_annotations,'note':'Chosen source annotations retained exactly; manual recovery masks are proposals pending review.'})
    # This proposal visibly retains water/glare, so numeric generation must not imply approval.
    shark='archive/train.X1/n01484850/n01484850_20518.JPEG'
    if shark in chosen:
        r=chosen[shark]
        save_json(report/'quality_review.json',{'config_sha256':sha((out/'config.json').read_bytes()),'images':{shark:{'decision':'reject','source_sha256':r['source_sha256'],'mask_sha256':r['file_hashes'][r['directory']+'/mask.png'],'reason':'Visual inspection of neutral composite: substantial water/glare retained; foreground boundary is unreliable. Requires corrected annotation/mask.'}}})
    summary=export_manifests(out,report,rows)
    summary['backend_source_counts']=counts;save_json(report/'status.json',summary)
    print('Merged source backends:',counts,flush=True)
    print('Running full independent pixel, split, label, ratio, provenance and Color-membership audit...',flush=True)
    result=audit(out,report)
    result['backend_source_counts']=counts;save_json(report/'audit.json',result)
    if not result['issues']:save_json(report/'status.json',{**summary,'status':'full_numeric_audit_passed_quality_review_pending','numeric_audit_passed':True,'training_ready':False})
    if os.environ.get('TEMPLATE2_BACKUP_DIR'):
        from colab_checkpoint import Backup
        print('Saving merged candidates, manifests, provenance and audit to Drive...',flush=True)
        Backup(ROOT,out).save(force=True)
    if result['issues']:raise RuntimeError('Full audit failed; see saved audit.json')
    print('FULL NUMERICAL AUDIT PASSED. Quality review is still pending; training_ready=False.',flush=True)

if __name__=='__main__':
    if sys.argv[1]=='--execute':execute(*sys.argv[2:])
    else:prepare(*sys.argv[1:])
