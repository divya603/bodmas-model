"""
test_recovery.py

Parameter-recovery check for the Bayesian inference model (inference.py).

We already know the ground-truth misconceptions behind every item in
stimulus_pool.json, since we generated them ourselves. This script samples
some of those items, runs posterior_over_profiles() on each trace using
*only* the trace (not the known answer), and checks whether the model's
top-ranked (MAP) profile matches the true one — the standard way to
validate an inference procedure before pointing it at real, unknown data.

Run from base-task/:
    python3 test_recovery.py
"""

import json
import random

from inference import posterior_over_profiles, most_likely_profile, format_profile

N_SAMPLE = 40
SEED = 0


def main():
    pool = json.load(open('stimulus_pool.json'))
    random.seed(SEED)
    sample = random.sample(pool, N_SAMPLE)

    hits = 0
    mass_on_truth = []

    for it in sample:
        true_profile = tuple(it['misconceptions'])  # already in canonical order
        post = posterior_over_profiles(it['trace'])
        map_profile, map_p = most_likely_profile(post)
        truth_p = post.get(true_profile, 0.0)
        mass_on_truth.append(truth_p)

        correct = (map_profile == true_profile)
        hits += correct
        tag = 'OK  ' if correct else 'MISS'
        print(f"{tag}  true={format_profile(true_profile):35s} "
              f"MAP={format_profile(map_profile):35s} "
              f"P(MAP)={map_p:.3f}  P(true)={truth_p:.3f}")

    print()
    print(f"MAP recovery: {hits}/{len(sample)}")
    print(f"avg posterior mass on true profile: {sum(mass_on_truth)/len(mass_on_truth):.3f}")


if __name__ == '__main__':
    main()
