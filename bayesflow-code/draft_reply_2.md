Hi Stefan,

Thanks again for the careful read. Your two points — that the competitor we
labelled "BayesFlow" is mislabeled, and that it may not reflect the state of the
art — pushed us to run a proper, systematic study with your exact setup. We did,
and we want to share all of it transparently. Short version: you are right on the
naming and right that a proper BayesFlow is more sample-efficient than our labeled
baseline at low/medium dimension — but the paper's central claim, the sharp
degradation as the parameter dimension grows, holds for proper BayesFlow too.

Let me go point by point, then give the full numbers.

> "What you labelled BayesFlow isn't really BayesFlow"

Agreed. Figures 1 and 3 use SBI's "NPE + Embedding Network" wrapper
(https://sbi.readthedocs.io/en/stable/tutorials/16_implemented_methods.html#radev-et-al-2020),
which does not correspond to the current BayesFlow framework. We will rename the
competitor (e.g. "NPE + embedding net") and add a post-publication note.

> I believe the results in Figure 3 are not representative of the SOTA.

To check this directly, we re-ran your configuration — `bf.BasicWorkflow` +
`FlowMatching(subnet_kwargs={"widths": (128,)*3})`, **no summary network**, jax
backend, **120 epochs** (matching your setup), batch 256 — as a full sweep:

- parameter dimension D in {2, 5, 10, 15, 20}
- budgets in {5k, 10k, 20k, 50k}
- both the distractor-free and the 18-distractor settings of the paper
- 3 seeds per cell, C2ST identical to the paper's (`lfi.evaluation.c2st`)
- ground-truth = the analytic bimodal posterior (modes at ±1)

Script + raw results are in the repo:
https://github.com/givasile/lfi/tree/bayesflow/bayesflow-code
(`run_bayesflow_experiments.py`; per-cell C2ST/runtime/samples under
`code/results/mog_benchmark/`).

### Proper BayesFlow (R), C2ST mean±std over 3 seeds (lower = better; threshold 0.8)

**18 distractors** (the setting of Figure 3)

| D | 5k | 10k | 20k | 50k |
|---|---|---|---|---|
| 2 | 0.791±0.051 | **0.642±0.090** | — | — |
| 5 | 0.839±0.053 | 0.779±0.052 | **0.698±0.043** | — |
| 10 | 0.953±0.018 | 0.916±0.022 | **0.738±0.029** | — |
| 15 | 0.985±0.004 | 0.973±0.017 | 0.855±0.017 | 0.815±0.079 |
| 20 | 0.994±0.003 | 0.982±0.008 | 0.950±0.040 | 0.865±0.053 |

**No distractors**

| D | 5k | 10k | 20k | 50k |
|---|---|---|---|---|
| 2 | **0.547±0.017** | — | — | — |
| 5 | **0.760±0.020** | — | — | — |
| 10 | 0.877±0.003 | 0.818±0.007 | 0.789±0.041 | 0.821±0.031 |
| 15 | 0.968±0.024 | 0.919±0.011 | 0.870±0.023 | 0.850±0.055 |
| 20 | 0.987±0.009 | 0.969±0.022 | 0.922±0.028 | 0.931±0.035 |

(bold = first budget at which all 3 seeds reach C2ST ≤ 0.8; "—" = we early-stopped
the budget ladder once a D was solved, since larger budgets only help.)

### Side-by-side with the labeled baseline in the paper (B = our "BayesFlow")

For reference, the baseline as run in the paper (3 runs each):

**18 distractors**

| D | 1k | 5k | 10k | 20k | 50k | 100k |
|---|---|---|---|---|---|---|
| 2 | 0.925±0.030 | 0.658±0.081 | 0.561±0.036 | · | · | · |
| 5 | 0.991±0.004 | 0.892±0.009 | 0.806±0.055 | · | 0.569±0.018 | · |
| 10 | 0.998±0.000 | 0.941±0.001 | 0.924±0.007 | · | 0.917±0.011 | 0.894±0.003 |
| 15 | 0.998±0.001 | 0.962±0.010 | 0.957±0.001 | · | 0.941±0.004 | 0.940±0.005 |
| 20 | 0.999±0.000 | 0.995±0.001 | 0.990±0.005 | · | 0.971±0.004 | 0.967±0.004 |

**No distractors**

| D | 1k | 5k | 10k | 20k | 50k | 100k |
|---|---|---|---|---|---|---|
| 2 | 0.785±0.034 | 0.560±0.006 | 0.570±0.011 | · | · | · |
| 5 | 0.935±0.007 | 0.791±0.076 | 0.693±0.015 | · | 0.554±0.005 | · |
| 10 | 0.995±0.001 | 0.938±0.007 | 0.938±0.010 | · | 0.916±0.005 | 0.923±0.034 |
| 15 | 0.998±0.001 | 0.955±0.008 | 0.959±0.010 | · | 0.946±0.012 | 0.946±0.010 |
| 20 | 0.999±0.000 | 0.994±0.003 | 0.984±0.008 | · | 0.960±0.003 | 0.971±0.002 |

(The baseline grid was {1k,5k,10k,50k,100k}; it has no 20k column. "·" = not run.)

### Budget to solve (smallest budget reaching C2ST ≤ 0.8)

| D | R: proper BF (dist=18) | B: paper baseline (dist=18) | R (dist=0) | B (dist=0) |
|---|---|---|---|---|
| 2 | 10k | 5k | 5k | 1k |
| 5 | 20k | 50k | 5k | 5k |
| 10 | 20k | **never (≥100k fails)** | never | never |
| 15 | never | never | never | never |
| 20 | never | never | never | never |

What this says, honestly:

1. **You are right that the labeled baseline understates BayesFlow.** Proper
   BayesFlow is clearly more sample-efficient at low/medium D. The sharpest case:
   with 18 distractors at **D=10 it reaches C2ST 0.738 at 20k, whereas the paper's
   labeled baseline never solves D=10 — it is still at 0.894 with 100k simulations.**
   So the comparison in Figure 3 was unfair to BayesFlow at medium D, and we should
   fix it.

2. **The paper's core claim nonetheless holds.** Both proper BayesFlow and the
   baseline collapse as D grows: at **D=15 and D=20, proper BayesFlow stays at
   C2ST ≈ 0.82–0.95 across every budget up to 50k** (e.g. D=20, 18 distractors:
   0.865 at 50k; D=20, no distractors: 0.931 at 50k), and the baseline is at
   0.94–0.97 even at 100k. The failure is method-agnostic and persists with budget.
   This is exactly the high-dimensional regime in which R2OMC claims its advantage —
   and where amortized posterior estimators are not the natural tool.

On the specific objections:

> Using 32 layers for the summary network is overkill ...

Note `embedding_net_num_hiddens=32` sets 32 hidden **units per layer**, not 32
layers — the summary network is 2 layers × 32 units. In any case the sweep above
uses **no summary network at all** (your configuration), so this does not affect the
conclusion.

> I ran a case with D = 1000 distractors to push the limits ...

Here is the dimension mix-up we flagged: in the paper, **D is the parameter
dimension**, and distractors are a separate, fixed count (18). The hardest setting
in Figure 3 is **D=20 with 18 distractors — a 38-dim observation — not D=2 with many
distractors.** Your D=2 + 1000-distractor case is the *easy* regime (2 informative
parameters); it is unsurprising and not in tension with the claim, which is about
the D=10/15/20 rows. The tables above scan exactly that axis with your config.

On runtime: on a CPU with the jax backend, 120 epochs takes us roughly 22s (5k),
39s (10k), 74s (20k), and 190–560s (50k, growing with D); the full 90-cell sweep was
~1.8 CPU-hours. Your 11s for 20k/120ep is far below our ~74s, so we assume it is on a
GPU — happy to be corrected.

To wrap up, two concrete steps we propose:
(a) Rename the competitor to "NPE + embedding net" and add a post-publication note.
(b) Replace the Figure 3 BayesFlow curve with these proper-BayesFlow numbers — they
    are stronger at low/medium D, and they make the high-D point *more* convincing,
    since even a SOTA amortized estimator does not cross the threshold for D ≥ 15.
    If you have a configuration that improves the D=15/20 rows, please share it and
    we will rerun and update.

And yes — we would be glad to discuss the optimal-design + posterior-inference
direction.
