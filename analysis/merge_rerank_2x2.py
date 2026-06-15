"""严格 2×2 merge × rerank 实验分析（qwen3.7-max）

读 4 个 judged.json，输出：
1. 2×2 总体表 + Δ
2. 分 Cat 详细表
3. 逐题 diff：rerank 主效应、merge 主效应、交互效应
4. 哪些问题被 rerank 修好/搞坏（merge ON vs OFF 分别列）

用法：python analysis/merge_rerank_2x2.py
"""
import json
import os
from collections import defaultdict

EXPERIMENTS = {
    'Y-max  (merge ON,  no rerank)': 'evaluation/results/expYmax_fs_k5_no_rr_merge_on_max/judged.json',
    'X-max  (merge OFF, no rerank)': 'evaluation/results/expXmax_fs_k5_no_rr_no_merge_max/judged.json',
    'Y-max-rr (merge ON,  +rerank)': 'evaluation/results/exp_ymax_rr/judged.json',
    'X-max-rr (merge OFF, +rerank)': 'evaluation/results/exp_xmax_rr/judged.json',
}


def load_labels(path):
    """读 judged.json → {(sample_id, question): {cat, gt, response, label}}"""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    out = {}
    for sid, items in data.items():
        for it in items:
            out[(sid, it['question'])] = {
                'cat': it.get('category', -1),
                'gt': it.get('gt_answer', ''),
                'response': it.get('response', ''),
                'label': it.get('llm_label', -1),
            }
    return out


def by_cat(data):
    """{(1,2,3,4): list of labels}"""
    bc = defaultdict(list)
    for v in data.values():
        bc[v['cat']].append(v['label'])
    return bc


def pct(lst):
    if not lst:
        return 0.0
    return sum(lst) / len(lst) * 100


def main():
    print('='*78)
    print('严格 2×2 merge × rerank 实验（qwen3.7-max）')
    print('='*78)

    data = {}
    for name, path in EXPERIMENTS.items():
        d = load_labels(path)
        if d is None:
            print(f'\n[WARN] {path} 不存在，跳过')
            continue
        data[name] = d
        n = len(d)
        bc = by_cat(d)
        overall = pct([v['label'] for v in d.values()])
        print(f'\n{name}: {n} 题, overall={overall:.1f}%')
        for c in [1, 2, 3, 4]:
            print(f'  Cat{c}: {pct(bc.get(c, [])):.1f}%  ({len(bc.get(c, []))} 题)')

    if len(data) < 4:
        print('\n[ERROR] 缺 judged.json，需要 4 个都跑完才能分析')
        return

    # 2×2 主表
    print('\n' + '='*78)
    print('2×2 主表（overall, qwen3.7-max）')
    print('='*78)
    Y_off = pct([v['label'] for v in data['Y-max  (merge ON,  no rerank)'].values()])
    Y_on = pct([v['label'] for v in data['Y-max-rr (merge ON,  +rerank)'].values()])
    X_off = pct([v['label'] for v in data['X-max  (merge OFF, no rerank)'].values()])
    X_on = pct([v['label'] for v in data['X-max-rr (merge OFF, +rerank)'].values()])
    print(f'{"":18} | {"rerank OFF":>12} | {"rerank ON":>12} | {"Δ rerank":>10}')
    print('-'*68)
    print(f'{"merge ON":18} | {Y_off:>11.1f}% | {Y_on:>11.1f}% | {Y_on-Y_off:>+9.1f}pp')
    print(f'{"merge OFF":18} | {X_off:>11.1f}% | {X_on:>11.1f}% | {X_on-X_off:>+9.1f}pp')
    print(f'{"Δ merge":18} | {Y_off-X_off:>+11.1f}pp | {Y_on-X_on:>+11.1f}pp |')

    # 分 Cat 详细
    print('\n' + '='*78)
    print('分 Category 详细')
    print('='*78)
    for cat in [1, 2, 3, 4]:
        print(f'\n--- Cat{cat} ---')
        y_off = pct([v['label'] for v in data['Y-max  (merge ON,  no rerank)'].values() if v['cat']==cat])
        y_on = pct([v['label'] for v in data['Y-max-rr (merge ON,  +rerank)'].values() if v['cat']==cat])
        x_off = pct([v['label'] for v in data['X-max  (merge OFF, no rerank)'].values() if v['cat']==cat])
        x_on = pct([v['label'] for v in data['X-max-rr (merge OFF, +rerank)'].values() if v['cat']==cat])
        print(f'  merge ON : rerank OFF={y_off:.1f}%, ON={y_on:.1f}%, Δ={y_on-y_off:+.1f}pp')
        print(f'  merge OFF: rerank OFF={x_off:.1f}%, ON={x_on:.1f}%, Δ={x_on-x_off:+.1f}pp')

    # 逐题 diff：rerank 主效应（在每个 merge 状态下）
    print('\n' + '='*78)
    print('逐题 diff：rerank 主效应')
    print('='*78)

    def diff(name_a, a, name_b, b, cat_filter=None):
        common = set(a) & set(b)
        if cat_filter:
            common = {k for k in common if a[k]['cat'] in cat_filter}
        a_yes = sum(1 for k in common if a[k]['label']==1)
        b_yes = sum(1 for k in common if b[k]['label']==1)
        aab = sum(1 for k in common if a[k]['label']==1 and b[k]['label']==1)
        anb = sum(1 for k in common if a[k]['label']==1 and b[k]['label']!=1)
        bna = sum(1 for k in common if a[k]['label']!=1 and b[k]['label']==1)
        n = len(common)
        return n, a_yes, b_yes, aab, anb, bna

    print('\n--- 在 merge ON 下，rerank 的净效应 (Y-max-rr vs Y-max) ---')
    n, ay, by, c, anb, bna = diff('Y-max', data['Y-max  (merge ON,  no rerank)'],
                                   'Y-max-rr', data['Y-max-rr (merge ON,  +rerank)'])
    print(f'  {n} 题共同；rerank OFF yes={ay}, ON yes={by}; 净增 {by-ay:+d}题 ({(by-ay)/n*100:+.1f}pp)')
    print(f'  rerank 杀对: {anb} 题 | rerank 救错: {bna} 题')
    if anb or bna:
        # 列出几个具体例子
        a_d = data['Y-max  (merge ON,  no rerank)']
        b_d = data['Y-max-rr (merge ON,  +rerank)']
        breaks = [k for k in a_d if k in b_d and a_d[k]['label']==1 and b_d[k]['label']!=1]
        fixes = [k for k in a_d if k in b_d and a_d[k]['label']!=1 and b_d[k]['label']==1]
        if breaks:
            print(f'  rerank 致败案例（前 3 个）：')
            for k in breaks[:3]:
                print(f'    [{k[0]}] {k[1][:60]}...')
                print(f'      GT: {a_d[k]["gt"]!r}')
                print(f'      OFF: {a_d[k]["response"]!r} → ON: {b_d[k]["response"]!r}')
        if fixes:
            print(f'  rerank 修复案例（前 3 个）：')
            for k in fixes[:3]:
                print(f'    [{k[0]}] {k[1][:60]}...')
                print(f'      GT: {a_d[k]["gt"]!r}')
                print(f'      OFF: {a_d[k]["response"]!r} → ON: {b_d[k]["response"]!r}')

    print('\n--- 在 merge OFF 下，rerank 的净效应 (X-max-rr vs X-max) ---')
    n, ay, by, c, anb, bna = diff('X-max', data['X-max  (merge OFF, no rerank)'],
                                   'X-max-rr', data['X-max-rr (merge OFF, +rerank)'])
    print(f'  {n} 题共同；rerank OFF yes={ay}, ON yes={by}; 净增 {by-ay:+d}题 ({(by-ay)/n*100:+.1f}pp)')
    print(f'  rerank 杀对: {anb} 题 | rerank 救错: {bna} 题')
    if anb or bna:
        a_d = data['X-max  (merge OFF, no rerank)']
        b_d = data['X-max-rr (merge OFF, +rerank)']
        breaks = [k for k in a_d if k in b_d and a_d[k]['label']==1 and b_d[k]['label']!=1]
        fixes = [k for k in a_d if k in a_d and a_d[k]['label']!=1 and b_d[k]['label']==1]
        if breaks:
            print(f'  rerank 致败案例（前 3 个）：')
            for k in breaks[:3]:
                print(f'    [{k[0]}] {k[1][:60]}...')
                print(f'      GT: {a_d[k]["gt"]!r}')
                print(f'      OFF: {a_d[k]["response"]!r} → ON: {b_d[k]["response"]!r}')
        if fixes:
            print(f'  rerank 修复案例（前 3 个）：')
            for k in fixes[:3]:
                print(f'    [{k[0]}] {k[1][:60]}...')
                print(f'      GT: {a_d[k]["gt"]!r}')
                print(f'      OFF: {a_d[k]["response"]!r} → ON: {b_d[k]["response"]!r}')

    # merge 主效应
    print('\n' + '='*78)
    print('逐题 diff：merge 主效应')
    print('='*78)
    print('\n--- 在 rerank OFF 下，merge 的净效应 (Y-max vs X-max) ---')
    n, ay, by, c, anb, bna = diff('Y-max', data['Y-max  (merge ON,  no rerank)'],
                                   'X-max', data['X-max  (merge OFF, no rerank)'])
    print(f'  {n} 题共同；merge ON yes={ay}, OFF yes={by}; 净增 {by-ay:+d}题 ({(by-ay)/n*100:+.1f}pp)')

    print('\n--- 在 rerank ON 下，merge 的净效应 (Y-max-rr vs X-max-rr) ---')
    n, ay, by, c, anb, bna = diff('Y-max-rr', data['Y-max-rr (merge ON,  +rerank)'],
                                   'X-max-rr', data['X-max-rr (merge OFF, +rerank)'])
    print(f'  {n} 题共同；merge ON yes={ay}, OFF yes={by}; 净增 {by-ay:+d}题 ({(by-ay)/n*100:+.1f}pp)')

    # 结论
    print('\n' + '='*78)
    print('结论')
    print('='*78)
    rerank_under_merge_on = Y_on - Y_off
    rerank_under_merge_off = X_on - X_off
    merge_under_no_rerank = X_off - Y_off
    merge_under_rerank = X_on - Y_on
    print(f'  Rerank 主效应 (merge ON 下) : {rerank_under_merge_on:+.1f}pp')
    print(f'  Rerank 主效应 (merge OFF 下): {rerank_under_merge_off:+.1f}pp')
    print(f'  Merge  主效应 (no rerank 下) : {merge_under_no_rerank:+.1f}pp (X-off  - Y-off)')
    print(f'  Merge  主效应 (rerank 下)    : {merge_under_rerank:+.1f}pp (X-on   - Y-on)')
    print(f'  交互效应  = ({rerank_under_merge_off - rerank_under_merge_on:+.1f})pp '
          f'OR ({merge_under_rerank - merge_under_no_rerank:+.1f})pp')


if __name__ == '__main__':
    main()
