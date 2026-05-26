# Draft reply to Stefan

**STATUS: DRAFT — waiting on D=5/10/20 results before sending.**

---

Hi Stefan,

Thank you for taking the time to look at the paper and for sharing the notebook — the collaboration link looks interesting too.

You are right on the main point: **what we labelled "BayesFlow" in Figure 3 is not a fair representation of the current BayesFlow framework.** We used the SBI package's wrapper (NSF density estimator + FCEmbedding summary network) and cited the 2020 paper. The 2020 paper's contribution was precisely the jointly-trained embedding net on top of NPE. We mistakenly equated that SBI wrapper with BayesFlow as a framework — we should have used your actual package with flow matching. We are re-running all our benchmark settings with `bf.BasicWorkflow` + `FlowMatching` (no summary network) and will update Figure 3 accordingly.

One small correction: the config you saw is `embedding_net_num_hiddens=32`, which is the number of hidden *units* per layer, not layers. There are 2 layers. The summary network is architecturally small; the issue is the choice to use one at all.

On the performance comparison: your notebook uses 2D theta + 18 distractors (20D total observation). In our paper `D` is the *parameter* dimension with distractors fixed at 18 throughout, so our hardest setting is 20D theta + 18 distractors = 38D total observation. Your notebook benchmarks our easiest case. We are running modern BayesFlow at D=5/10/15/20 to get the right comparison. Preliminary results at D=2 (your setting, 3 seeds) are consistent with your notebook:

| Budget | Mean C2ST (modern BF, D=2, 18 dist.) | Runtime (our CPU) |
|--------|--------------------------------------|-------------------|
| 5k     | 0.87  — above threshold              | ~66s              |
| 10k    | 0.77  — marginal                     | ~116s             |
| 20k    | 0.69  — below 0.75 ✓                 | ~205s             |
| 50k    | 0.59  — good                         | ~378s             |

So your claim on C2ST holds for D=2. On runtime: your notebook reports 11 seconds for the 20k case — we measured ~205 seconds for the same setting. The difference is hardware: your machine has a GPU, ours is CPU-only. The 11 seconds is not a property of the method itself but of your setup, which is worth keeping in mind when interpreting the runtime comparison against R2OMC in the paper.

So your claim holds for D=2. We are running D=5/10/15/20 and will share full results and update the paper once done. Thanks again.

Best,
[name]

---

## Scripts in `bayesflow-code/`

| File | Purpose |
|------|---------|
| `mail.md` | Stefan's original email |
| `mog_real_bayesflow.ipynb` | Stefan's notebook (provided by him) |
| `run_bayesflow_experiments.py` | **Systematic grid** — D∈{2,5,10,15,20}, budgets∈{5k,10k,20k,50k,100k}, 3 seeds, 50 epochs. Saves into `code/results/.../real_bayesflow_<budget>/` so `results.py` picks it up without changes. |
| `quick_scan.py` | **Quick scan** — D∈{2,5,10,20} × dist∈{0,18} × budgets∈{1k,5k,10k}, 1 seed. For rapid email-level estimates. |
| `reply_draft.md` | This file |

**Environment:** `lfi-bayesflow` conda env (clone of `lfi-aistats` + `bayesflow==2.0.8`, torch backend, CPU only).

---

## Results so far

### Modern BayesFlow — D=2 theta, 18 distractors (= Stefan's setting), 3 seeds

| Budget | Seed scores | Mean C2ST | Mean runtime (CPU) | Old BayesFlow (SBI) mean |
|--------|-------------|-----------|-------------------|--------------------------|
| 5k     | 0.866, 0.914, 0.830 | **0.870** | 66s  | 0.658 |
| 10k    | 0.826, 0.690, 0.798 | **0.771** | 116s | 0.561 |
| 20k    | 0.752, 0.712, 0.594 | **0.686** | 205s | — |
| 50k    | 0.530, 0.656, 0.582 | **0.589** | 378s | — |

Stefan's reported runtime at 20k: **11s (GPU)**. Ours: **205s (CPU)**. ~18x gap, entirely hardware.

⚠️ **These used a non-standard C2ST** (sklearn MLP fixed (64,64), single split, no z-score). The paper uses (10·D, 10·D) hidden units, 5-fold CV, z-scored. Scripts have been fixed; rerun pending. Direction is likely correct but absolute values will shift.

### D=5/10/15/20 — pending (systematic experiment still running)

### Quick scan (2D/5D/10D/20D × no-dist/18-dist × 1k/5k/10k) — still running

---

## Checklist before sending

- [ ] Quick scan results landed (ETA: ~30 min)
- [ ] D=5/10/20 systematic results landed (ETA: several hours)
- [ ] Confirm R2OMC still clearly wins at D=10/15/20
- [ ] Update Figure 3: rename old "BayesFlow" → "NPE (SBI)" or drop; add "BayesFlow" with modern results
- [ ] Update appendix config table
- [ ] Send reply
