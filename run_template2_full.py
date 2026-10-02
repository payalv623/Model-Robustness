"""Generate every frozen source, audit it, and report honest release status."""
import json
import traceback
from background_dataset import run, paths, utc_now
from audit_background_dataset import audit
from background_progress_report import render as render_progress
from color_dataset import ROOT, save_json


def update_pipeline(state, **fields):
    path=ROOT/'project_metadata/pipeline_status.json'
    metadata=json.loads(path.read_text())
    metadata['templates']['background']=state
    metadata.update(fields,updated_utc=utc_now())
    save_json(path,metadata)


def main():
    output,report=paths(True)
    update_pipeline('full_candidate_generation_running',background_generation_status=str((report/'status.json').relative_to(ROOT)))
    try:
        run(full=True)
        update_pipeline('full_numeric_audit_running')
        result=audit(output,report)
        state=('complete_verified' if result['full_dataset_verified'] else
               'generated_audit_failed' if result['issues'] else 'full_candidates_generated_quality_review_pending')
        update_pipeline(state,background_numeric_audit=str((report/'audit.json').relative_to(ROOT)),
                        background_generated_sources=result['source_images_verified'],
                        background_unresolved_masks=result['quality_review_unresolved'])
        save_json(report/'job_outcome.json',{'status':state,'training_ready':result['training_ready'],
                  'updated_utc':utc_now(),'source_images':result['source_images_verified'],
                  'unresolved_masks':result['quality_review_unresolved'],
                  'visual_sample_pending':result['required_visual_sample_pending']})
        render_progress(report,{**result,'status':state,'total_sources':13500,'updated_utc':utc_now()})
        print('Template 2 job outcome:',state,flush=True)
    except Exception as exc:
        update_pipeline('full_generation_stopped_with_error')
        save_json(report/'job_outcome.json',{'status':'stopped_with_error','error':str(exc),'training_ready':False,'updated_utc':utc_now()})
        traceback.print_exc()
        raise


if __name__=='__main__':main()
