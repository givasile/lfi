Hi Stefan,

Thanks for the careful read.
On the naming side, your critique is fair; we probably mislabeled it.
On the experimental side, it is interesting to see if your framework/setup,
(`bf.BasicWorkflow + FlowMatching`, no summary network)
produces results that overturn the paper's claims.
On that, I think there is a mix-up between parameter dimension and observation dimension.

Let me go point by point.

> "What you labelled BayesFlow isn't really BayesFlow"

Agreed. Figures 1 and 3 practically use SBI's "NPE + Embedding Network" wrapper (https://sbi.readthedocs.io/en/stable/tutorials/16_implemented_methods.html#radev-et-al-2020),
which probably does not correspond to the latest  BayesFlow framework (?)
We can rename the competitor and add a short post-publication note clarifying that.

> I believe the results presented in Figure 3 are not representative of the SOTA in the field (even if it's what one gets when one uses the SBI package).

Figure 3 compares the main SBI methods using fixed defaults, i.e., same config across all D.
We spent some time tuning (not exhaustive hyperparam search) and did not find simple configurations that performed significantly better.
If you are aware of any SBI configs that clearly improve the three baselines we compare against, please share them and we will rerun and update the results.

> For example, using 32 layers for a summary network is an overkill for a problem that doesn't need a summary network ..

Note that `embedding_net_num_hiddens=32` sets 32 hidden units per layer, not 32 layers.
The summary network is 2 layers × 32 units.

On the distractor case, the summary net is meant to compress the informative dimensions and discard the rest.
We even sized its output to match the number of informative dimensions (D per setup), giving it the cleanest possible signal.
Why would you call it an overkill?

On the non-distractor case, we agree the summary net adds nothing.
We kept it for consistency with the distractor runs, expecting it to be roughly neutral rather than helpful.

> Additionally, I ran a case with D = 1000 distractors to push the limits of what bayesflow ...

Here is the mix-up I mentioned at the top: in our paper, D is the parameter dimension, and distractors are a separate, fixed count (18).
The hardest setting in Figure 3 is D=20 with 18 distractors, a 38-dim observation, not D=2 with 18 distractors.

Your notebook benchmarks D=2 + 18 distractors, which is the easiest row in that figure. The R2OMC claim is supported by the D=10/15/20 rows, not D=2.

To check how your setup scales across D, we re-ran a quick scan with your config  `bf.BasicWorkflow + FlowMatching(widths=(128,)*3)`, no summary network;
You may check the script here: https://github.com/givasile/lfi/blob/bayesflow/bayesflow-code/quick_scan.py

  D    distractors    budget=1k    budget=5k    budget=10k
  2         0           0.816        0.617 ✓      0.582 ✓
  2        18           0.928        0.815        0.823
  5         0           0.946        0.843        0.844
  5        18           0.988        0.924        0.895
  10        0           0.994        0.948        0.876
  10       18           0.995        0.982        0.948
  20        0           1.000        0.997        0.996
  20       18           1.000        0.998        0.996

- D=2 + 18 distractors: 0.823 at 10k/50ep on CPU. Your 20k/120ep on GPU gets below 0.75; same direction, so your D=2 result holds.
- D=5 + 18 distractors: 0.895. D=10: 0.948. D=20: ~1.0 across the whole budget range. The picture changes as D grows, which is the regime the paper's claim is about.

A small side question on runtime: your 11s for 20k/120ep, is it on a GPU?
Same training on my CPU takes ~200s, and I'd like to understand the gap.

On the D=1000 run:
same point as above; D in the paper is the parameter dimension, not the distractor count.
Interesting however that the network handles 1000 distractors well at D=2.

To wrap up, two steps seem fair:
(a) Rename the competitor and add a post-publication note.
(b) Re-run the Figure 3 experiments using the BayesFlow framework and see where it lands. If you can share a notebook or a proposed config for doing so, it would be great.

And yes, we would be happy to discuss the optimal design + posterior inference direction.
