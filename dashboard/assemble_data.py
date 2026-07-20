#!/usr/bin/env python3
"""
assemble_data.py

Build dashboard/dashboard_data.json: for every pool item, the stimulus fields +
the Bayesian ideal-observer marginal + all three LLM regimes (rating, correct,
and haiku-thinking's reasoning trace) + the anonymized human practice-cohort
responses. PII-free by construction (no Prolific IDs, seedIDs, or demographics;
participants are relabeled P01..Pn).

Prereqs:
  dashboard/bayes_per_item.json   (run the block at the bottom once, or it is
                                    recomputed here if missing)
  llm_exp/results/raw_{haiku_thinking,haiku_direct,gpt4o_direct}_all487.jsonl
  data/real-all-main-data.json    (gitignored; raw pull)

Run from repo root:
    python3 dashboard/assemble_data.py
"""
import json
import os
import sys
from statistics import NormalDist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(ROOT, 'dashboard')
z = NormalDist().inv_cdf


def bayes_per_item():
    path = os.path.join(HERE, 'bayes_per_item.json')
    if os.path.exists(path):
        return json.load(open(path))
    sys.path.insert(0, os.path.join(ROOT, 'base-task'))
    from inference import posterior_over_profiles, marginal_rule_probability
    pool = json.load(open(os.path.join(ROOT, 'base-task/stimulus_pool.json')))
    out = {it['id']: round(marginal_rule_probability(
        posterior_over_profiles(it['trace']), it['probed_misconception']), 4) for it in pool}
    json.dump(out, open(path, 'w'))
    return out


def load_llm(name):
    seen = {}
    for line in open(os.path.join(ROOT, 'llm_exp/results', name)):
        r = json.loads(line)
        if 'all_items' in r['subject_id'] and r.get('response') is not None:
            seen[r['id']] = r
    return seen


def correct(rating, sc):
    return (rating >= 4) == (sc is True)


def task_trials(d):
    return [t for t in d.get('pageData_exp', {}).get('visit_0', {}).get('data', [])
            if isinstance(t, dict) and 'response' in t]


def is_complete(d):
    return d.get('done') is True and len(task_trials(d)) >= 24 and d.get('recruitmentService') == 'prolific'


def main():
    pool = json.load(open(os.path.join(ROOT, 'base-task/stimulus_pool.json')))
    bayes = bayes_per_item()
    thinking = load_llm('raw_haiku_thinking_all487.jsonl')
    hdirect = load_llm('raw_haiku_direct_all487.jsonl')
    gpt4o = load_llm('raw_gpt4o_direct_all487.jsonl')

    recs = [r['data'] for r in json.load(open(os.path.join(ROOT, 'data/real-all-main-data.json')))]
    practice = sorted([d for d in recs if d.get('pageData_practice') and is_complete(d)],
                      key=lambda d: d.get('starttime', {}).get('_seconds', 0))

    item_human = {it['id']: [] for it in pool}
    participants = []
    for k, d in enumerate(practice, 1):
        pid = f"P{k:02d}"
        trials = task_trials(d)
        ncorr = sum(correct(t['response'], t['statement_correct']) for t in trials)
        sig = [t for t in trials if t['category'] in ('A', 'C')]
        noi = [t for t in trials if t['category'] in ('B', 'D')]

        def rate(s):
            n = len(s); p = sum(t['response'] >= 4 for t in s) / n
            return min(max(p, 1 / (2 * n)), 1 - 1 / (2 * n))
        hit, fa = rate(sig), rate(noi)
        bycat = {}
        for c in 'ABCD':
            s = [t for t in trials if t['category'] == c]
            bycat[c] = round(sum(correct(t['response'], t['statement_correct']) for t in s) / len(s), 3) if s else None
        sp = d.get('pageData_strategy', {}).get('visit_0', {}).get('data', [])
        strat = sp[0].get('strategy', '') if (sp and isinstance(sp, list) and isinstance(sp[0], dict)) else ''
        participants.append({'pid': pid, 'n': len(trials), 'acc': round(ncorr / len(trials), 3),
                             'dprime': round(z(hit) - z(fa), 2), 'crit': round(-0.5 * (z(hit) + z(fa)), 2),
                             'byCat': bycat, 'strategy': strat})
        for t in trials:
            item_human[t['id']].append({'p': pid, 'r': t['response']})

    items = []
    for it in pool:
        iid, sc = it['id'], it['statement_correct']
        th, hd, gp = thinking.get(iid), hdirect.get(iid), gpt4o.get(iid)
        hr = item_human[iid]
        items.append({
            'id': iid, 'cat': it['category'], 'nm': it['num_misconceptions'],
            'expr': it['expression'], 'trace': it['trace'], 'stmt': it['belief_statement'],
            'name': it['student_name'], 'mis': it['misconceptions'], 'probed': it['probed_misconception'],
            'sc': sc, 'wt': it.get('which_target'), 'fs': it.get('foil_status'), 'bayes': bayes.get(iid),
            'llm': {
                'thinking': {'r': th['response'], 'ok': correct(th['response'], sc),
                             'tok': th.get('thinking_tokens'), 'reason': th.get('reasoning', '')} if th else None,
                'hdirect': {'r': hd['response'], 'ok': correct(hd['response'], sc)} if hd else None,
                'gpt4o': {'r': gp['response'], 'ok': correct(gp['response'], sc)} if gp else None,
            },
            'human': {'n': len(hr), 'nagree': sum(1 for x in hr if x['r'] >= 4),
                      'ncorr': sum(1 for x in hr if (x['r'] >= 4) == (sc is True)), 'resp': hr},
        })

    out = {'meta': {'n_items': len(items), 'n_participants': len(participants),
                    'regimes': ['thinking', 'hdirect', 'gpt4o'],
                    'note': 'BODMAS 480-pool dashboard; human = practice cohort (anonymized)'},
           'items': items, 'participants': participants}
    json.dump(out, open(os.path.join(HERE, 'dashboard_data.json'), 'w'))
    print(f"wrote dashboard_data.json: {len(items)} items, {len(participants)} participants")


if __name__ == '__main__':
    main()
