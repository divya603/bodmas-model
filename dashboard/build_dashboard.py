#!/usr/bin/env python3
"""
build_dashboard.py

Inject the assembled dataset and the thinking-trace failure analysis into
template.html, producing a self-contained index.html to publish as an Artifact.

  dashboard/dashboard_data.json      <- stimuli + bayes + 3 LLM regimes + humans
  dashboard/thinking_failures.json   <- {per_item:[...], summary:{...}} (optional)
  dashboard/template.html            <- page shell with __DATA__ / __FAIL__ slots
  -> dashboard/index.html

Run from repo root:
    python3 dashboard/build_dashboard.py
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def embed(obj):
    # compact JSON, made safe to sit inside <script type="application/json">
    return json.dumps(obj, separators=(',', ':'), ensure_ascii=False).replace('</', '<\\/')


def reconcile_failures(fail, item_ids):
    """Keep only failures for items still in the pool (dropping e.g. removed
    ambiguous items) and recompute the summary counts + headline numbers so the
    thinking-errors view stays consistent with the item table."""
    orig = fail.get('per_item', [])
    kept = [f for f in orig if f['id'] in item_ids]
    if len(kept) == len(orig):
        return fail
    newc = Counter(f['category'] for f in kept)
    o_fa = sum(f.get('direction') == 'false-agree' for f in orig)
    o_fd = sum(f.get('direction') == 'false-disagree' for f in orig)
    n_fa = sum(f.get('direction') == 'false-agree' for f in kept)
    n_fd = sum(f.get('direction') == 'false-disagree' for f in kept)
    head = fail.get('summary', {}).get('headline', '')
    head = head.replace(f"The {len(orig)} errors", f"The {len(kept)} errors")
    head = head.replace(f"all {o_fa} false-agrees", f"all {n_fa} false-agrees")
    head = head.replace(f"all {o_fd} false-disagrees", f"all {n_fd} false-disagrees")
    for c in fail['summary']['categories']:
        head = head.replace(f"{c['slug']} ({c['count']})", f"{c['slug']} ({newc[c['slug']]})")
        c['count'] = newc.get(c['slug'], 0)
    fail['summary']['categories'] = [c for c in fail['summary']['categories'] if c['count']]
    fail['summary']['headline'] = head
    fail['per_item'] = kept
    return fail


def main():
    data = json.load(open(os.path.join(HERE, 'dashboard_data.json')))
    fpath = os.path.join(HERE, 'thinking_failures.json')
    fail = json.load(open(fpath)) if os.path.exists(fpath) else {'per_item': [], 'summary': {}}
    fail = reconcile_failures(fail, {it['id'] for it in data['items']})

    tpl = open(os.path.join(HERE, 'template.html')).read()
    html = tpl.replace('__DATA__', embed(data)).replace('__FAIL__', embed(fail))
    out = os.path.join(HERE, 'index.html')
    with open(out, 'w') as f:
        f.write(html)
    mb = os.path.getsize(out) / 1e6
    print(f"wrote {out}  ({mb:.2f} MB)  items={len(data['items'])} "
          f"participants={len(data['participants'])} failures={len(fail.get('per_item', []))}")


if __name__ == '__main__':
    main()
