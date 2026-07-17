# human_buffer

Intermediate (throwaway) human plots for the NEW practice-trial task on the 480 pool.
Only the new-task cohort (participants with a `pageData_practice` block), not the old
24-person pilot. Regenerate after each `npm run getdata` pull:

```
python3 analysis_human/plot_human_1misc_distributions.py --cohort practice --out human_buffer
python3 analysis_human/plot_2misc_heatmap.py            --cohort practice --out human_buffer
```

Outputs: human_1misc_dist_A.png, human_1misc_dist_B.png, human_2misc_heatmap_practice.png

Cohort discriminator = `pageData_practice` present (timestamp-independent; the old pilot
has none). Refuted split for dist_B row 3 comes from analysis-Bayesian/b_item_marginals.json
(regenerated on the 480 pool). If performance improves, run a bigger batch and promote the
finalized figures to Results_combined/figs.
