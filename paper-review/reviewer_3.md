# Answer to Reviewer 3

We sincerely thank the reviewer for the insightful comments, especially the ones about posterior coverage, ESS and the high-dimensional image experiment. We have performed additional analyses to fully address the raised points, where possible.

---

**Significance Justification:**

R: Concern over small $\epsilon$ -- "With small ε ... is problematic."

A: The reviewer's concern is that a small $\epsilon$ may lead to a restricted posterior spread. We acknowledge this concern but clarify that the spread is primarily driven by simulator noise (see point 1 below) and the multiple seed stratagy (see point 2 below), not solely by $\epsilon$ threshold. 

To clarify:

1. $\epsilon$ constrains in data space, not parameter space.

   The threshold $\epsilon$ acts on the distance between simulated data points $d(x, x_{obs})$, not the distance in the parameter space $d(\theta, \theta_{true})$). A small $\epsilon$ can still yield a broad region if the simulator exhibits low sensivity to changes around $\theta_{true}$. Therefore, although it is true that decreasing $\epsilon$ will decrease the (per-seed) proposal region, this is not done in a linear manner.

2. Muliple seeds naturally increase coverage.

   For each observation we run $��$ seeds, each producing its own proposal region. Even if each seed-specific region is individually tight, simulator randomness ensures they to land in different parts of the parameter space. Their union naturally expands the overall posterior support.

Furthermore, the reviewer correctly notices the absence of (global or local) coverage diagnostics; unfortunately, these are prohibitively expensive for non-amortized methods like R2OMC as they require generating a large calibration set of prior samples and associated simulations and performing inference **for every calibration point** (Deistler et al., Simulation-Based Inference: A Practical Guide).

However, we have performed Posterior Predictive Checks (PPCs). These checks confirm that the distribution of simulated data generated from the approximate posterior is consistent with the observed data, demonstrating that our posterior does not exhibit major failure modes. 
Specifically, in PPCs, for a given observation(s), we draw $N$ posterior samples and simulate one output for each. We then verify that the observation is consistent with distribution of these simulated outputs. A well-calibrated posterior should reproduce the observation within simulator noise (as is the case in SLCP). These PPC results will be included in our camera-ready Appendix.

---

R: "For SLCP the autho ... no C2ST scores."

A:  We have calculated the following efficiency statistics:

- Overall Efficiency: Our approach yields an overall efficiency of approximately $\approx 5 $ per cent. This means we generate about $\mathbf{20}$ proposal samples for every $\mathbf{1}%$ final retained posterior sample.

- Sample Generation Breakdown: 
  - In a typical run, we generate $20,000$ total proposals (4 observations by 5,000 each)
  - Approximately 4,000 ($\approx 20$ per cent) are initially accepted (receive a positive weight), and, finally, we resample from this weighted pool to retain 1,000 final posterior samples.
  - We then resample from this weighted pool to retain 1,000 final posterior samples.

- Effective Sample Size (ESS): The ESS depends on the particular run and simulation budget. We have performed some experiments that show that, on a typical run, the ESS of the weighted pool of accepted samples is $\approx 1,400$, which confirms that the weights are not degenerate and the pool contains sufficient independent information to support the target size of posterior draws.

We will include precise statistics for the above metrics in the revised manuscript. 

---

R: "Unfortunately, for the high dimensional image setting there are no baselines and no C2ST scores."

A: Thank you for your valuable feedback regarding the need for quantitative metrics. We agree that including appropriate metrics is essential for a thorough evaluation.

- Baselines:

  We acknowledge the absence of common baselines (like NPE or BayesFlow) for the high-dimensional image setting. Our initial exploratory experiments indicated that these methods struggle significantly with raw high-dimensional outputs, likely requiring extensive pre-processing, dimensionality reduction, or specialized architectures to perform competitively. Working on these modifications is beyond the scope of the current work. We therefore opted to focus on directly demonstrating R2OMC's capability in effectively handling high-dimensional data under limited simulation budgets, rather than presenting potentially misleadingly poor baseline performance.

- C2ST: We apologize for the omission of the C2ST scores for the first version of the high-dimensional image experiment. As the simulator applies an affine transformation + noise, computing the ground truth posterior is indeed feasible. We have now computed these scores using 1000 generated samples (from the ground truth and the approximate posterior) across 10 distinct observations. The resulting C2ST score is $0.52 \pm 0.03$. We will ensure this metric is included in the revised manuscript. For the second version, it is not so straightforward to obtain ground truth posterior due the checkerboard filter.

---

R: "The paper is well written ... in their experiments."

A: We thank the reviewer for the positive feedback on the clarity of the paper and our discussion of its limitations. In our experiments, these limitations did **not** materially affect R2OMC's performance.

Specifically:

- Limitation 1 -- Requirement for differentiable simulators:
  All simulators used in our experiments were intentionally chosen to be differentiable, so this constraint did not hinder performance.

- Limitation 2 -- Performance depends on successful optimization:
  Gradient-based optimization was generally successful in our tests, and we did not observe major issues. Nevertheless, local minima or non-convergence can occur and would negatively affect R2OMC' performance. We therefore recommend that practitioners always inspect the distances at the optimization endpoints: if many $d^*$ values do not approach zero, this is a clear signal that optimization issues need to be addressed.

- Limitation 3 -- ε may grow large to cover all observations:
  In SLCP, $\epsilon$ remained well-controlled. Observation-based proposal regions (per seed and observation) were typically built with ε≈0.5 and samples drawn from them were tested to "match" with all observations. The threshold to retain a positive weight was set to about ε≈2.0. If the distance with respect to each observation (at least one seed-specific distance per observation) was below that threshold, then they were weighted following Eq. 14. So, $\epsilon$ practical and stable in our runs.

- Limitation 4 -- Optimization does not explicitly incorporate the prior.:
  This can push proposal regions into low- or zero-prior areas. In the 2-moons simulator, for example, we observe that some endpoints $\theta_i^*$ fell outside the prior support. However, because a sufficient number of endpoints remained within the prior mass, inference accuracy was preserved, and overall performance was not meaningfully affected. Note that, when this limitation does not harm performance, it can be viewed as a feature: we can easily change the prior without expensive further compute and thus easily perform prior sensitivity analysis.

---

R: "The labels and legends of some figures are very small."

A: We have noticed that and we will fix it.
