import sys
sys.path.insert(0, '.')
from batch_judge import judge_group

judge_group(
    'outputs/g14_lw_v2_l40_mergeoff/neatmem_results.json',
    'outputs/g14_lw_v2_l40_mergeoff/judged_rerun.json'
)
