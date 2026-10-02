"""Plan assignments without claiming that segmented images have been generated."""

from collections import Counter
from fractions import Fraction

from color_dataset import ROOT, IDS, SPLITS, load_sources, plan_conditions, atomic_bytes, csv_bytes, save_json, sha


def prepare_plan(root=ROOT):
    rows, metadata, split_config = load_sources(root)
    assignments = plan_conditions(rows, {'correlation': '9/10', 'seed': 42})
    mapping = {'correlated': 'correlated', 'balanced': 'randomized', 'reversed': 'reversed',
               'counterfactual': 'counterfactual', 'all_nature': 'all_red', 'all_urban': 'all_blue'}
    records = []
    for row in rows:
        for condition, color_condition in mapping.items():
            category = {'red': 'nature', 'blue': 'urban_indoor'}[assignments[row['path']][color_condition]]
            majority = 'nature' if row['group'] == 'A' else 'urban_indoor'
            records.append({'source': row['path'], 'source_sha256': row['sha256'],
                            'class_id': row['class_id'], 'class_index': row['class_index'],
                            'group': row['group'], 'split': row['split'], 'condition': condition,
                            'planned_background': category, 'aligned': category == majority,
                            'generation_status': 'planned_only'})
    distributions = {}
    for split in SPLITS:
        for condition in mapping:
            for class_id in IDS:
                selected = [r for r in records if r['split'] == split and r['condition'] == condition and r['class_id'] == class_id]
                n = split_config['per_class_counts'][class_id][split]
                aligned = sum(r['aligned'] for r in selected)
                expected = {'correlated': int(n * Fraction(9,10)), 'balanced': n//2,
                            'reversed': n-int(n * Fraction(9,10)), 'counterfactual':0}
                if len(selected) != n or (condition in expected and aligned != expected[condition]):
                    raise ValueError('Background assignment count mismatch')
                distributions[f'{split}/{condition}/{class_id}'] = {'total':n, 'aligned':aligned,
                          'backgrounds':dict(Counter(r['planned_background'] for r in selected))}
    directory = root / 'reports/background_bias_current'
    content = csv_bytes(records, tuple(records[0]))
    atomic_bytes(directory / 'assignment_plan.csv', content)
    report = {'status':'assignments_verified_not_generated', 'source_images':len(rows),
              'planned_condition_records':len(records), 'correlation':.9,
              'same_frozen_splits_as_color':True, 'same_aligned_minority_source_membership_as_color':True,
              'source_split_manifest_sha256':sha((root/'project_metadata/source_splits.csv').read_bytes()),
              'assignment_plan_sha256':sha(content),'distributions':distributions,
              'controls_planned':['untouched RGB','same selected foreground on neutral gray'],
              'limitations':['Synthetic nature and urban scenes differ in color and structure; compound appearance shift.',
                             'SAM automatic masks are class-agnostic; target identity must be checked.',
                             'Assignment verification is not segmentation or dataset completion.']}
    save_json(directory / 'assignment_plan_audit.json', report)
    print(f'Background assignment plan verified: {len(rows)} sources, {len(records)} planned records; no production images generated.', flush=True)
    return report


if __name__ == '__main__':
    prepare_plan()
