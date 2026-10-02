"""Regenerate/resume Shape candidates, then audit; visual approval is never fabricated."""
from color_dataset import ROOT
from shape_dataset import generate,visual_sample
from audit_shape_dataset import audit

if __name__=='__main__':
    out=generate()
    report=ROOT/'reports/shape_bias_current'
    visual_sample(ROOT,out,report)
    result=audit(ROOT,out,report)
    raise SystemExit(1 if result['issues'] else 0)
