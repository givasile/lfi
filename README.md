# Fast and Robust Simulation-Based Inference With Optimization Monte Carlo

**Vasilis Gkolemis, Christos Diou, Michael U. Gutmann**

[Paper (arXiv)](https://arxiv.org/abs/2511.13394) | [lfi package (master branch)](https://github.com/givasile/lfi)

---

We propose **R2OMC** (Robust and Rapid Optimization Monte Carlo), a simulation-based inference method that is fast, robust to distractor dimensions, and applicable to multi-modal posteriors.

## Repository Structure

```
arXiv-submission/   - LaTeX source of the arXiv submission
camera-ready/       - LaTeX source of the AISTATS 2026 camera-ready version
code/               - Experiment code and lfi package (see code/README.md)
bayesflow-code/     - Post-publication BayesFlow follow-up experiments (see below)
paper-review/       - Reviewer comments and author responses
```

## Reproducing the Experiments

See [`code/README.md`](code/README.md) for setup and step-by-step instructions.

## Using R2OMC

To use R2OMC in your own work, see the [`master` branch](https://github.com/givasile/lfi).

## Citation

```bibtex
@inproceedings{gkolemis2026r2omc,
  title     = {Fast and Robust Simulation-Based Inference With Optimization Monte Carlo},
  author    = {Gkolemis, Vasilis and Diou, Christos and Gutmann, Michael U.},
  booktitle = {Proceedings of the 29th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year      = {2026}
}
```

## Additional BayesFlow Experiments (post-publication)

Following Stefan Radev (BayesFlow author) question on whether our neural baseline 
reflected the **current BayesFlow** framework, we re-ran the modern BayesFlow stack
on the paper's main benchmark (the 20D Gaussian Mixture). 
The results are added in the appendix.

**Takeaways.** Modern BayesFlow is more sample-efficient than the originally
reported baseline at low/medium dimension. However, accuracy still degrades
sharply as the parameter dimension grows: for **D ≥ 15** it stays at
**C2ST ≈ 0.82–0.95 across every budget up to 50k simulations**,
confirming the paper's main claim that neural posterior estimators struggle to 
scale with the parameter dimension at low-to-mid budgets.

The sweep script is [`bayesflow-code/run_bayesflow_experiments.py`](bayesflow-code/run_bayesflow_experiments.py);
per-cell raw C2ST / runtime / posterior samples are stored under
`code/results/mog_benchmark/<setting>/D_<D>/real_bayesflow_<budget>/`.


