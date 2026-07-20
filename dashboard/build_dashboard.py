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

HERE = os.path.dirname(os.path.abspath(__file__))


def embed(obj):
    # compact JSON, made safe to sit inside <script type="application/json">
    return json.dumps(obj, separators=(',', ':'), ensure_ascii=False).replace('</', '<\\/')


def main():
    data = json.load(open(os.path.join(HERE, 'dashboard_data.json')))
    fpath = os.path.join(HERE, 'thinking_failures.json')
    fail = json.load(open(fpath)) if os.path.exists(fpath) else {'per_item': [], 'summary': {}}

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
