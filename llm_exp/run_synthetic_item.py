#!/usr/bin/env python3
"""
run_synthetic_item.py

Run the synthetic sub<÷-refutable B item (analysis-Bayesian/synthetic_items.json,
NOT part of the deployed pool) through the three LLM regimes used for the main
runs (haiku thinking / haiku direct / gpt-4o direct), with identical determinism
settings (temperature 0, top_p 1, seed 0, practice=none, independent delivery).
Writes the ratings back into synthetic_items.json under `llm_ratings`.

Run from llm_exp/:
    python3 run_synthetic_item.py
"""

import json
import os

from bodmas_llm.bodmas_prompt import RESPONSE_FORMAT, build_messages
from bodmas_llm.client import OpenRouterClient

HERE = os.path.dirname(os.path.abspath(__file__))
SYNTH = os.path.join(os.path.dirname(HERE), 'analysis-Bayesian', 'synthetic_items.json')

REGIMES = [
    ('haiku (thinking)', 'anthropic/claude-haiku-4.5', 'thinking', 16000),
    ('haiku (direct)', 'anthropic/claude-haiku-4.5', 'direct', 8192),
    ('gpt-4o (direct)', 'openai/gpt-4o', 'direct', 8192),
]


def main():
    items = json.load(open(SYNTH))
    for item in items:
        print(f"{item['id']}  present={item['misconceptions'][0]}  "
              f"foil={item['probed_misconception']}  "
              f"bayes marginal={item.get('bayes_marginal_on_probe')}")
        messages = build_messages(item, worked_examples=False)
        ratings = {}
        for label, model, effort, max_tokens in REGIMES:
            client = OpenRouterClient(
                model=model,
                params={'temperature': 0.0, 'top_p': 1.0, 'seed': 0,
                        'max_tokens': max_tokens})
            resp = client.call(messages, response_format=RESPONSE_FORMAT, effort=effort)
            rating = (resp.answer_json or {}).get('rating')
            ratings[label] = rating
            print(f"  {label:18s} rating={rating}  "
                  f"(thinking_tokens={resp.reasoning_tokens}, cached={resp.cached}"
                  + (f", error={resp.error}" if resp.error else '') + ')')
        item['llm_ratings'] = ratings

    with open(SYNTH, 'w') as f:
        json.dump(items, f, indent=2)
    print(f"\nRatings saved into {SYNTH}")


if __name__ == '__main__':
    main()
