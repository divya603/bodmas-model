#!/usr/bin/env python3
"""
make_bonus_list.py

Produce a Prolific bulk-bonus list ("prolific_id,amount" per line) from a
downloaded Smile study.

Inputs (defaults match `npm run getdata` / `npm run getrecruitment` output):
  --data         data/real-all-main-data.json          (participant records)
  --recruitment  data/private/real-main-recruitment.json (session_id -> prolific_id)

For each COMPLETED (done) session recruited via Prolific, the bonus is
recomputed here from the participant's raw Likert responses against the
embedded answer key (statement_correct), NOT trusted from the client-stored
value — client data is tamperable. The 6-point scale is collapsed to a binary
agree/disagree (response >= 4 = agree) and scored against the item's
ground-truth direction; the bonus is rescaled so chance (50%) earns $0 and a
perfect run earns --max-bonus, rounded to cents:

    bonus = max(0, (accuracy - 0.5) / 0.5) * max_bonus

Sessions whose recomputed bonus disagrees with the client-stored value, or
completed Prolific sessions with no matching prolific_id, are reported to
stderr so nothing is silently mispaid or dropped.

Already-paid participants are tracked in a ledger (default
data/private/bonus_paid.csv, gitignored) and skipped on later runs, so the
output only ever contains people you have NOT yet bonused. Workflow:

    python3 scripts/make_bonus_list.py              # list unpaid only
    # ... paste into Prolific's bulk-bonus box and pay ...
    python3 scripts/make_bonus_list.py --mark-paid  # record them in the ledger

--mark-paid is idempotent (re-marking already-ledgered people is a no-op) and
--include-paid prints everyone regardless of the ledger, for auditing.

Usage:
    python3 scripts/make_bonus_list.py
    python3 scripts/make_bonus_list.py --exclude <seedID_or_prolificID>,<...>
    python3 scripts/make_bonus_list.py --out data/private/bonuses.csv
"""

import argparse
import csv
import json
import os
import sys
from datetime import date

MAX_BONUS = 2.0
CHANCE = 0.5
LEDGER = 'data/private/bonus_paid.csv'
LEDGER_FIELDS = ['prolific_id', 'amount', 'seed', 'marked_on']


def load_ledger(path):
    """prolific_id -> row dict for everyone already paid."""
    if not os.path.exists(path):
        return {}
    with open(path, newline='') as f:
        return {row['prolific_id']: row for row in csv.DictReader(f)}


def append_ledger(path, rows):
    new_file = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if new_file:
            w.writeheader()
        w.writerows(rows)


def recompute_bonus(trials, max_bonus, chance):
    scored = [t for t in trials
              if isinstance(t, dict) and 'response' in t and 'statement_correct' in t]
    n = len(scored)
    correct = sum(1 for t in scored
                  if (t['response'] >= 4) == (t['statement_correct'] is True))
    accuracy = correct / n if n else 0.0
    bonus = round(max(0.0, (accuracy - chance) / (1 - chance)) * max_bonus, 2)
    return n, correct, accuracy, bonus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/real-all-main-data.json')
    ap.add_argument('--recruitment', default='data/private/real-main-recruitment.json')
    ap.add_argument('--out', default=None, help='write the list to a file as well as stdout')
    ap.add_argument('--exclude', default='', help='comma-separated seedIDs or prolific_ids to drop (e.g. your own test runs)')
    ap.add_argument('--max-bonus', type=float, default=MAX_BONUS)
    ap.add_argument('--chance', type=float, default=CHANCE)
    ap.add_argument('--ledger', default=LEDGER, help='CSV of already-paid participants (skipped on output)')
    ap.add_argument('--mark-paid', action='store_true',
                    help='record the participants listed by this run in the ledger')
    ap.add_argument('--include-paid', action='store_true',
                    help='list everyone, ignoring the ledger (audit mode; incompatible with --mark-paid)')
    args = ap.parse_args()

    if args.include_paid and args.mark_paid:
        ap.error('--include-paid and --mark-paid cannot be combined '
                 '(would re-ledger already-paid participants)')

    exclude = {x.strip() for x in args.exclude.split(',') if x.strip()}
    paid = load_ledger(args.ledger)

    data = json.load(open(args.data))
    rec = json.load(open(args.recruitment))
    sid_to_pid = {r['session_id']: r['prolific_id'] for r in rec if r.get('service') == 'prolific'}

    records = [r['data'] for r in data]
    lines = []
    payees = []

    for d in records:
        if d.get('done') is not True or d.get('recruitmentService') != 'prolific':
            continue
        seed = d.get('seedID')
        pid = sid_to_pid.get(seed)

        if pid is None:
            print(f"WARN: completed Prolific session {seed} has no matching prolific_id "
                  f"in recruitment data — skipped", file=sys.stderr)
            continue
        if seed in exclude or pid in exclude:
            print(f"INFO: excluding {pid} (seed {seed[:8]})", file=sys.stderr)
            continue

        exp = d.get('pageData_exp', {}).get('visit_0', {}).get('data', [])
        trials = [t for t in exp if isinstance(t, dict) and 'response' in t]
        n, correct, accuracy, bonus = recompute_bonus(trials, args.max_bonus, args.chance)

        if pid in paid and not args.include_paid:
            prior = paid[pid]
            if abs(float(prior['amount']) - bonus) > 0.005:
                print(f"WARN: {pid} already paid {prior['amount']} on {prior['marked_on']} "
                      f"but now recomputes to {bonus:.2f} — check manually", file=sys.stderr)
            else:
                print(f"INFO: {pid} already paid {prior['amount']} on {prior['marked_on']} — skipped",
                      file=sys.stderr)
            continue

        bblock = [t for t in exp if isinstance(t, dict) and t.get('phase') == 'traceJudgmentBonus']
        client_bonus = bblock[0]['bonus'] if bblock else None
        if client_bonus is not None and abs(client_bonus - bonus) > 0.005:
            print(f"WARN: {pid} client-stored bonus {client_bonus} != recomputed {bonus} "
                  f"(using recomputed)", file=sys.stderr)

        if n != 24:
            print(f"WARN: {pid} has {n} scored trials (expected 24)", file=sys.stderr)

        lines.append(f"{pid},{bonus:.2f}")
        payees.append({'prolific_id': pid, 'amount': f'{bonus:.2f}',
                       'seed': seed, 'marked_on': date.today().isoformat()})

    out_text = '\n'.join(lines)
    print(out_text)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(out_text + '\n')
        print(f"\n({len(lines)} rows written to {args.out})", file=sys.stderr)

    if args.mark_paid:
        if payees:
            append_ledger(args.ledger, payees)
        print(f"({len(payees)} participants marked paid in {args.ledger}; "
              f"ledger now has {len(paid) + len(payees)} entries)", file=sys.stderr)


if __name__ == '__main__':
    main()
