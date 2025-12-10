We are pleased that the reviewer found the methodological construction coherent and recognized the expanded empirical evaluation. 

Our response focuses on demonstrating that R2OMC contributions are not merely incremental, but essential innovations that rendering the ROMC framework applicable to challenging, high-dimensional, and multi-observation problems.

The core idea of (R)OMC, recasting inference as an optimization problem, is very powerful.
Yet this advantage cannot be seen in the algorithm's current form. Our essential innovations render this powerful idea functional. We communicate these evidence-backed improvements, so that this approach is not overshadowed by the field's focus on major new algorithms.

---

**1. Summary and Contributions:**

R: "Firstly, I would like ... structure and content."

A: We acknowledge the similarities in foundational ideas with the prior submission. 
However, this version incorporates significant improvements driven by last year's feedback.
The previous submission was noted for its promising methodological framework and the criticism mainly focused on the experimental section. The key spotted limitations where:
 - (i) missing important baselines (e.g. Flow Matching, BayesFlow) and/or inadequate use of them (e.g., not tested under many budgets)
 - (ii) lack of enough and diverse experimental settings (missing high-dimensional problems and multiple observations settings).

In this version, we:

- **Expanded baselines and correct use (addressing point (i))**: 

    In Section 4.1 and in the concept figure, we have now included three established neural-based methods (NPE, BayesFlow, and Flow Matching), compared to only NPE in the previous version. We test each of these baselines across a wide range of simulation budgets (1K to 50K) repeatedly (3 to 5 independent runs), to make sure that our main claim (neural-based methods oftrn require a large simulation budget (>50K) whereas R2OMC is competitive with <1K) is secured.
  
- **Diverse experimental settings (addressing point (ii))**: 

  We have included two new settings:
  (a) Multiple observations and distractors (Section 4.2) in a well known and challenging SBI problem (SLCP and SLCP-distractors) and (b) a high-dimensional image-based simulator (Section 4.3).
  
- **Limitations Discussion**: A new limitations section (Section 3.4) to explicitly discuss the method's boundaries and inform under which conditions our approach may struggle.

We believe these enhancements address the previous experimental shortcomings.

---

**2. Significance Justification:**

R: "The empirical section... slightly limited"

A: We appreciate the recognition of our expanded empirical section. The reviewer raises a concern that most experiments are **based on synthetic or toy examples**.

About the concern of *toy examples*, we argue that all experiments (except from the first row of the concept figure) are **challenging** problems, mainly taken from established benchmarks in the SBI literature.
The fact that SotA competitors often struggle confirms their non-trivial nature. Section 4.1 (which is designed by us) uses a MoG benchmark that although uses a simple simulator, due to high-dimensionality, bi-modality and presence of distractors, it is quite challenging for most SBI methods, as shown in Figure 3. Section 4.2 uses three well-known and very challenging problems (SLCP, SLCP-distractors and 2-moon problem) from the established SBI benchmark. Section 4.3 uses a high-dimensional image-based simulator, where most SBI methods struggle due to the high-dimensionality of the data.

About the concern of *synthetic examples*, we argue that this is a common choice in the SBI literature, as synthetic benchmarks allow for known ground-truth posteriors, which are fundamental for an objective (quantitative) evaluation of SBI methods.  However, as we acknowledge that a real-world application would expand the practical relevance of our work, we are willing to test our method against a differentiable real-world simulator and include the results in the final version if accepted.

---

**3. Novelty Justification:**

R: "The methodological contribution is an incremental extension of the established Robust Optimization Monte Carlo (ROMC) framework": 

A: While we build upon the ROMC framework, our additions, i.e., (a) gradient-based optimization and jax-based implementation, (b) handling distractors and (c) our approach on iid observations, are not merely incremental. We clarify that standard ROMC fails in all our experimental scenarios. In Sections 4.1 and 4.3, it fails due to absence of gradients and not taking advantage of jax speedups (fails to converge in reasonable time). In the distractors-version of Section 4.1, it fails due to the presence of distractors. In Section 4.2 , it fails due to both distractors and multiple observations. Therefore, our approach does not just slightly improve ROMC; it renders ROMC applicable to a broad class of challenging problems that were previously out of reach.

R: "The proposed handling ... ROMC itself."
    
A: We agree that learning summary statistics is a viable alternative. However, this approach introduces significant engineering overhead, requires large simulation budgets for training, and adds complexity, hyperparameters, and potential instability.
Our methodological contribution is that we sidestep this challenge by leveraging the problem's structure (i.i.d observations) to decompose it into a set of one-observation sub-problems that are easier to solve. Therefore, the user can perform inference under multiple observations without the need for additional work (training, engineering) and with minimal computational overhead (linear in the number of observations).

R: "Filtering distractor ... or inferential mechanism."

A: The simplicity of the mechanism (simple thresholding rule is), enables efficient identification and exclusion of uninformative dimensions *without significant computational overhead or complexity*.  
We acknowledge that setting the threshold introduces a hyperparameter that may require tuning. 

However, we note that:

- Using machine-precision (a very small value) serves as a robust default, filtering out only the dimensions that have absolutely no influence on the output.

- Prior knowledge of the simulator allows users to set a higher value, selectively filtering dimensions with very low, but non-zero, influence to potentially increase the method's accuracy .

To fully address this point, we will include a dedicated ablation study in the Appendix to thoroughly demonstrate the effect of this threshold hyperparameter on the method's performance.
