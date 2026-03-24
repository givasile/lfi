Paper Decision
Decisionby Program Chairs22 Jan 2026, 14:23 (modified: 22 Jan 2026, 14:55)Program Chairs, Senior Area Chairs, Area Chairs, AuthorsRevisions
Decision: Accept (Poster)
Comment:
After a constructive discussion, the reviewers have raised their scores significantly (3, 5, 1 (withdrawn), 3 => then 6, 6, 5). New experiments and results have been provided by the authors, clarifications have been given, and concerns have been addressed to the satisfaction of the reviewers. For these reasons, I follow the reviewers' recommendations and recommend acceptance. I encourage the authors to address the last remaining reviewers' comments (if any) and to implement the modifications/clarifications discussed with the reviewers in the final version of the paper.

Official Comment by Area Chair 7fyh
Official Commentby Area Chair 7fyh08 Dec 2025, 22:43Program Chairs, Senior Area Chairs, Area Chairs, Reviewers, Authors
Comment:
Dear authors and reviewers,

Thank you for engaging in the reviewing process so far.

The current ratings of 3, 5, 1, 3 show that there is strong disagreement among the reviewers, even after considering Reviewer 4f6x's wish to withdraw their review.

@Reviewers: Please make sure to read the authors' rebuttals carefully and consider comments from other reviewers. Make sure you now have all the necessary information to make an informed and fair assessment of the paper. Make sure to summarize your recommendation clearly in your review. You may also update your scores and edit your initial review if you wish to do so.

@Authors: Within the remaining time-frame, please feel free to respond to any additional comments or questions from the reviewers. Note that reviewers may also update their scores and edit their initial review. Please refrain from responding too extensively, as the focus should be on clarifying any misunderstandings or answering specific questions.

Sincerely, The AC

General response
Official Commentby Authors (Christos Diou, Vasileios Gkolemis, Michael U. Gutmann)28 Nov 2025, 13:20 (modified: 10 Dec 2025, 15:33)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers, Reviewers Submitted, AuthorsRevisions
Comment:
We appreciate the reviewers’ recognition of our core contributions.

All reviewers noted that our method achieves accurate posterior inference under a limited simulation budget and welcomed our handling of distractors and i.i.d. observations.

The provided feedback was centered around three axes: (a) incremental contribution, (b) limited experiments, and (c) constructive suggestions (adding clarifications: e.g. ESS, relative work: e.g. SimFormer, BOLFI).

We addressed each of these points carefully in our per-reviewer responses. Briefly, for (a) we clarified that our work is not incremental over previous methods as it makes (R)OMC applicable to a wide range of problems previously out of reach. For (b) we emphasized that our benchmarks were already challenging and provided additional empirical evidence. For (c) we welcomed the suggestions and will incorporate them in the camera-ready version.

We are happy that our responses satisfactorily addressed all their concerns, as reflected in the very positive updated scores. We thank the reviewers for their thoughtful feedback.

Below, we emphasize a general point.

(R)OMC [1], [2] presents a unique perspective to SBI due to converting likelihood-free inference to deterministic optimization. This approach offers significant advantages, which mainly stem from its ability to efficiently navigate in high-dimensional parameter spaces. Importantly, these advantages complement existing neural-based approaches, which excel in terms of accuracy but require extensive simulation budgets.

We believe the true potential of (R)OMC is underappreciated within the SBI community, as in its current form it is limited to simple applications. Our work (gradient-based optimization, handling distractors, and ensuring IID assumptions) unlocks the full potential of this method.

We stress that it is important to communicate these evidence-backed findings so that this unique SBI-perspective is not overshadowed by the field's focus on major new algorithms.

[1] Meeds, Ted, and Max Welling. "Optimization Monte Carlo: Efficient and embarrassingly parallel likelihood-free inference." Advances in Neural Information Processing Systems 28 (2015).

[2] Ikonomov, Borislav, and Michael U. Gutmann. "Robust optimisation monte carlo." International Conference on Artificial Intelligence and Statistics. PMLR, 2020.

Official Review of Submission1393 by Reviewer M3rS
Official Reviewby Reviewer M3rS11 Nov 2025, 11:33 (modified: 13 Dec 2025, 00:33)Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer M3rSRevisions
Summary And Contributions:
Firstly, I would like to note that I previously reviewed a submission for AISTATS 2025 which presented an extremely similar methodological contribution. While I cannot confirm whether this manuscript was authored by the same group, there is a high degree of overlap in terms of ideas, structure and content. The paper addresses Bayesian inference for complex stochastic simulators with intractable likelihoods. As in the earlier work I reviewed, the method builds on the Robust Optimisation Monte Carlo (ROMC) framework and proposes an approach referred to as R2OMC that combines deterministic optimization with gradient information to identify regions of high posterior density. The paper continues to target the same two key challenges: handling multiple independent and identically distributed (iid) observations and mitigating the adverse effect of distractor output dimensions. Compared to the submission I reviewed in 2025, the experimental section has been expanded.

Paper Keywords: Simulation-Based Inference, Optimization Monte Carlo, Stochastic Optimization
Expertise Keywords: Bayesian Inference, Approximate Bayesian Computation, Monte Carlo Methods
Soundness: Correct / minor errors (e.g., typo or errors that do not affect the main results)
Soundness Justification:
The methodological construction is coherent and correctly grounded in the existing ROMC framework. The optimisation-based reformulation, use of deterministic seeds and gradient-based procedures are technically sound and consistent. As the paper does not present any new theoretical results or formal guarantees, the focus of soundness is on the accuracy of the algorithmic workflow and the empirical methodology. The experimental design is well structured with multiple baselines and repeated runs, and the evaluation metrics are appropriate for comparing likelihood-free inference methods. However, although the experiments are carefully executed, they remain confined to synthetic settings, which limits the assessment of soundness in more complex or noisy real-world contexts. Overall, the paper is technically sound within its scope, but more substantial empirical validation is needed to establish robustness fully.

Significance: Somewhat significant (e.g., theoretical contribution of limited novelty or empirical gains missing baselines/statistical rigor).
Significance Justification:
The empirical section presents a wide range of comparisons, including several modern neural-network-based SBI methods (NPE, BayesFlow, Flow Matching) evaluated across different parameter dimensions and simulation budgets. This expanded set of baselines strengthens the experimental part of the paper and helps to show how the proposed approach fits into the current SBI landscape.

At the same time, most of the experiments are based on synthetic or toy examples. These settings clearly show the method's intended strengths, but they don't show how it would actually work in practice. The work shows that the approach can be competitive in controlled synthetic scenarios, but it is still difficult to assess how effective it is on more complex or realistic simulators. So, the wider practical importance of the contribution is slightly limited.

Novelty: Incremental compared to existing results
Novelty Justification:
After the rebuttal of the authors and the additional simulation results on the Lotka–Volterra model have convinced me that my initial review was somewhat harsh. I also acknowledge that I judged the incremental character of the work too severely, partly because I remained too influenced by the impression left by last year’s version.

Before the rebuttal :

The methodological contribution is an incremental extension of the established Robust Optimization Monte Carlo (ROMC) framework. The proposed handling of iid observations relies on optimising each observation independently and mixing the resulting proposal distributions. While this is a workable solution, it raises conceptual questions, since the difficulty appears more closely related to the design of appropriate summary statistics or distance functions than to a structural limitation of ROMC itself.

Filtering distractor dimensions through gradient-norm sensitivity remains the method's most original aspect. However, the approach remains relatively simple, being based on a thresholding rule, and does not introduce a fundamentally new modelling or inferential mechanism.

Although the manuscript includes significantly expanded empirical comparisons, particularly with modern, neural-based SBI methods, these additions primarily strengthen the experimental evaluation and do not alter the underlying methodological novelty, which remains incremental.

Non-conventional Contributions:
The paper does not introduce unconventional research contributions.

Clarity:
The paper remains well written. The exposition has been reorganised but still requires improved reference formatting.

Relation To Prior Work: All related works are clearly discussed
Reproducibility: Sufficient amount of details available for reproducing the main results
Rating: 6: Accept (Technically solid paper, with high impact on at least one sub-area. The contribution is convincing and the evaluation is adequate, though there may be some limitations or open questions.)
Confidence: 5: Absolutely certain (Use this sparingly; please ensure you are very familiar with the related work, you have carefully checked and understood the proofs and/or experimental details if applicable)
Rebuttal by Authors
Rebuttalby Authors (Christos Diou, Vasileios Gkolemis, Michael U. Gutmann)27 Nov 2025, 20:24 (modified: 01 Dec 2025, 15:15)Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewers SubmittedRevisions
Rebuttal:
We are pleased that the reviewer found the methodological construction coherent and recognized the expanded empirical evaluation.

Our response focuses on demonstrating that R2OMC contributions are not merely incremental, but essential innovations that rendering the ROMC framework applicable to challenging, high-dimensional, and multi-observation problems.

The core idea of (R)OMC, recasting inference as an optimization problem, is very powerful. Yet this advantage cannot be seen in the algorithm's current form. Our essential innovations render this powerful idea functional. We communicate these evidence-backed improvements, so that this approach is not overshadowed by the field's focus on major new algorithms.

1. Summary and Contributions:

R: "Firstly, I would like ... structure and content."

A: We acknowledge the similarities in foundational ideas with the prior submission. However, this version incorporates significant improvements driven by last year's feedback. The previous submission was noted for its promising methodological framework and the criticism mainly focused on the experimental section. The key spotted limitations where:

(i) missing important baselines (e.g. Flow Matching, BayesFlow) and/or inadequate use of them (e.g., not tested under many budgets)
(ii) lack of enough and diverse experimental settings (missing high-dimensional problems and multiple observations settings).
In this version, we:

Expanded baselines and correct use (addressing point (i)):

In Section 4.1 and in the concept figure, we have now included three established neural-based methods (NPE, BayesFlow, and Flow Matching), compared to only NPE in the previous version. We test each of these baselines across a wide range of simulation budgets (1K to 50K) repeatedly (3 to 5 independent runs), to make sure that our main claim (neural-based methods oftrn require a large simulation budget (>50K) whereas R2OMC is competitive with <1K) is secured.

Diverse experimental settings (addressing point (ii)):

We have included two new settings: (a) Multiple observations and distractors (Section 4.2) in a well known and challenging SBI problem (SLCP and SLCP-distractors) and (b) a high-dimensional image-based simulator (Section 4.3).

Limitations Discussion: A new limitations section (Section 3.4) to explicitly discuss the method's boundaries and inform under which conditions our approach may struggle.

We believe these enhancements address the previous experimental shortcomings.

2. Significance Justification:

R: "The empirical section... slightly limited"

A: We appreciate the recognition of our expanded empirical section. The reviewer raises a concern that most experiments are based on synthetic or toy examples.

About the concern of toy examples, we argue that all experiments (except from the first row of the concept figure) are challenging problems, mainly taken from established benchmarks in the SBI literature. The fact that SotA competitors often struggle confirms their non-trivial nature. Section 4.1 (which is designed by us) uses a MoG benchmark that although uses a simple simulator, due to high-dimensionality, bi-modality and presence of distractors, it is quite challenging for most SBI methods, as shown in Figure 3. Section 4.2 uses three well-known and very challenging problems (SLCP, SLCP-distractors and 2-moon problem) from the established SBI benchmark. Section 4.3 uses a high-dimensional image-based simulator, where most SBI methods struggle due to the high-dimensionality of the data.

About the concern of synthetic examples, we argue that this is a common choice in the SBI literature, as synthetic benchmarks allow for known ground-truth posteriors, which are fundamental for an objective (quantitative) evaluation of SBI methods. However, as we acknowledge that a real-world application would expand the practical relevance of our work, we are willing to test our method against a differentiable real-world simulator and include the results in the final version if accepted.

3. Novelty Justification:

R: "The methodological contribution is an incremental extension of the established Robust Optimization Monte Carlo (ROMC) framework":

A: While we build upon the ROMC framework, our additions, i.e., (a) gradient-based optimization and jax-based implementation, (b) handling distractors and (c) our approach on iid observations, are not merely incremental. We clarify that standard ROMC fails in all our experimental scenarios. In Sections 4.1 and 4.3, it fails due to absence of gradients and not taking advantage of jax speedups (fails to converge in reasonable time). In the distractors-version of Section 4.1, it fails due to the presence of distractors. In Section 4.2 , it fails due to both distractors and multiple observations. Therefore, our approach does not just slightly improve ROMC; it renders ROMC applicable to a broad class of challenging problems that were previously out of reach.

R: "The proposed handling ... ROMC itself."

A: We agree that learning summary statistics is a viable alternative. However, this approach introduces significant engineering overhead, requires large simulation budgets for training, and adds complexity, hyperparameters, and potential instability. Our methodological contribution is that we sidestep this challenge by leveraging the problem's structure (i.i.d observations) to decompose it into a set of one-observation sub-problems that are easier to solve. Therefore, the user can perform inference under multiple observations without the need for additional work (training, engineering) and with minimal computational overhead (linear in the number of observations).

R: "Filtering distractor ... or inferential mechanism."

A: The simplicity of the mechanism (simple thresholding rule is), enables efficient identification and exclusion of uninformative dimensions without significant computational overhead or complexity.
We acknowledge that setting the threshold introduces a hyperparameter that may require tuning.

However, we note that:

Using machine-precision (a very small value) serves as a robust default, filtering out only the dimensions that have absolutely no influence on the output.

Prior knowledge of the simulator allows users to set a higher value, selectively filtering dimensions with very low, but non-zero, influence to potentially increase the method's accuracy .

To fully address this point, we will include a dedicated ablation study in the Appendix to thoroughly demonstrate the effect of this threshold hyperparameter on the method's performance.

Follow-up on our rebuttal
Official Commentby Authors (Christos Diou, Vasileios Gkolemis, Michael U. Gutmann)06 Dec 2025, 20:41 (modified: 06 Dec 2025, 20:44)Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewers SubmittedRevisions
Comment:
Dear Reviewer M3rS,

Thank you again for your constructive review. We would like to kindly check whether any points remain unclear or would benefit from further clarification. Since the discussion period ends in about three days, we would greatly appreciate any feedback you might have, so that we can address any remaining concerns in time.

Kind regards,

The Authors

Follow-up on rebuttal with an additional experiment
Official Commentby Authors (Christos Diou, Vasileios Gkolemis, Michael U. Gutmann)09 Dec 2025, 13:09 (modified: 09 Dec 2025, 14:09)Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewers SubmittedRevisions
Comment:
To further address the reviewer’s concern regarding applicability to a more realistic model, we conducted an additional analysis on the Lotka-Volterra model, a standard problem in the SBI literature [1]. The Lotka-Volterra model is characterized by its use of an internal system of differential equations (DE) to generate a sequence of outputs as observations, in a time-series manner. Using an underlying system of DE is a common approach for modeling natural phenomena through a simulator but, so far, we had not tested our method against such a case.

Problem Description (Full specification details are provided in [1]):

Prior: 

α ∼ LogNormal(−0.125, 0.5)
β ∼ LogNormal(−3, 0.5),
γ ∼ LogNormal(−0.125, 0.5),
δ ∼ LogNormal(−3, 0.5)
Simulator: 
, where:

X, Y are simulated from:
 
 

Dimensionality: 
, 

Fixed parameters: 
, 

Implementation Note

To ensure compatibility with our JAX-based framework, we implemented the Lotka-Volterra simulator directly in JAX. Since our differential equation solver slightly varies from the Julia-based implementation used in [1], we cannot leverage the ground truth samples provided by that benchmark for a direct 
 comparison. Therefore, we validate our method through Simulation-Based Calibration (SBC), a standard approach for assessing posterior quality in simulation-based inference without ground truth samples [2].

Evaluation Metrics

We conducted Simulation-Based Calibration (SBC) diagnostics across 150 independent repetitions with 100 posterior samples each, utilizing a budget of 1,000 simulator calls per repetition. SBC assesses the statistical correctness of the approximate posterior 
 through two primary checks [2]:

Uniformity Test (SBC 
 and p-value)
Description: We first generate a parameter 
 from the prior, 
, and a corresponding observation 
 from the simulator, 
. We then sample 
 times from the approximate posterior, 
. For each parameter dimension 
, the rank of the generating parameter 
 is computed relative to the set of 
 posterior samples.

Expected Outcome: When the posterior is correctly calibrated, the resulting ranks (aggregated over all 
 replications) should be approximately uniformly distributed. The 
 statistic and its associated p-value test the null hypothesis that the observed rank distribution is uniform. A high p-value (as observed in our results) indicates correct calibration.

Empirical Coverage
Description: For each parameter dimension 
 and a desired nominal coverage level (e.g., 50%, 90%, 95%), we compute the corresponding central credible interval from the approximate posterior samples.

Expected Outcome: The empirical coverage is the fraction of the 150 replications where the true generating parameter 
 lies within the computed credible interval. Correct calibration requires that the empirical coverage closely matches the nominal coverage level (e.g., a 90% credible interval should contain the true parameter approximately 90% of the time).

Results - Table 1

Parameter	SBC χ²	SBC p-value	Coverage 50%	Coverage 90%	Coverage 95%
α (prey birth rate)	97.79	0.544	0.500	0.900	0.953
β (predation rate)	84.32	0.870	0.500	0.927	0.940
γ (predator death rate)	81.63	0.910	0.527	0.907	0.960
δ (reproduction efficiency)	89.71	0.760	0.480	0.907	0.987
Interpretation:

SBC uniformity tests: All p-values > 0.54 indicate no evidence of miscalibration (null hypothesis: correct calibration), with rank distributions consistent with uniformity.
Empirical coverage: Credible intervals achieve coverage matching nominal levels across all parameters (e.g., 90% intervals achieve 90.0–92.7% empirical coverage), demonstrating well-calibrated uncertainty quantification.
These results, obtained within a budget of 1,000 simulator calls, enhance the evidence of R2OMC's applicability to real-world problems which involve complex simulators with differential equations. While a direct 
 comparison against the benchmark's pre-computed ground truth is not available, the above calibration results are strong indication of an accurate posterior estimation by R2OMC.

Representative SBC rank histograms and coverage calibration plots will be included in the revised manuscript.

[1] Lueckmann, Jan-Matthis, et al. "Benchmarking simulation-based inference." International conference on artificial intelligence and statistics. PMLR, 2021. [2] Hermans, Joeri, et al. "A trust crisis in simulation-based inference? your posterior approximations can be unfaithful." arXiv preprint arXiv:2110.06581 (2021).

Official Comment by Reviewer M3rS
Official Commentby Reviewer M3rS09 Dec 2025, 20:06 (modified: 09 Dec 2025, 20:11)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer M3rS, AuthorsRevisions
Comment:
The rebuttal of the authors and the additional simulation results on the Lotka–Volterra model have convinced me that my initial review was somewhat harsh. I also acknowledge that I judged the incremental character of the work too severely, partly because I remained too influenced by the impression left by last year’s version. In light of the authors clarifications and improvements, I would update my rating to 6 (accept). I thank the authors for their efforts and for the quality of their responses.

 Replying to Official Comment by Reviewer M3rS
Response to Reviewer M3rS
Official Commentby Authors (Christos Diou, Vasileios Gkolemis, Michael U. Gutmann)10 Dec 2025, 10:44Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:
Thank you for your thoughtful reconsideration and we are glad the clarifications and new results addressed your concerns.

Official Review of Submission1393 by Reviewer 4f6x
Official Reviewby Reviewer 4f6x11 Nov 2025, 03:08 (modified: 13 Dec 2025, 00:33)Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer 4f6xRevisions
Summary And Contributions:
The core intent of the paper is unclear. Please work on presenting the main idea as well as your innovation over the existing literature more clear.

Paper Keywords: simulation, inference, monte carlo
Expertise Keywords: stochastic simulation, bayesian inference, stochastic optimization, mcmc
Soundness: Major errors (e.g., an incorrect theorem or derivation)
Soundness Justification:
The intent of this paper, its methodology, and the numerical experiments it claims to have conducted are not clear.

Significance: Not significant (e.g., theoretical results are incremental, or empirical results are similar to baselines/missing key comparisons).
Significance Justification:
A bunch of known definitions and results are introduced and then claimed to be part of something new.

Novelty: Known results, or a trivial extension of known results
Novelty Justification:
A bunch of known definitions and results are introduced and then claimed to be part of something new.

Non-conventional Contributions:
n/a

Clarity:
the paper keeps presenting ideas and then jumping away to something new without putting all the pieces it introduces together into a well formed approach to tackle a well defined problem.

Relation To Prior Work: Some important related works are missing or described incorrectly
Reproducibility: Insufficient amount of details available
Rating: 1: Strong Reject (For instance, a paper with well-known results or unaddressed ethical considerations.)
Confidence: 2: Somewhat confident (You are willing to defend your assessment, but it is quite likely that you did not understand central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.)
Rebuttal by Authors
Rebuttalby Authors (Christos Diou, Vasileios Gkolemis, Michael U. Gutmann)27 Nov 2025, 20:34 (modified: 01 Dec 2025, 15:15)Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewers SubmittedRevisions
Rebuttal:
We are disappointed by the reviewer's assessment, particularly given the complete absence of specific feedback or concrete examples to support the claims of major errors, lack of novelty, and unclear methodology.

Regarding clarity of intent and methodology

The manuscript clearly positions itself within the well-established domain of Simulation-Based Inference (SBI), explicitly building upon the Robust Optimization Monte Carlo (ROMC) framework. Section 3 provides detailed methodological descriptions, including uninformative dimensions (Section 3.1), iid observations (Section 3.2), jax gradient-based optimization (Section 3.3).

Regarding the experimental evaluation

Our experiments (Section 4) include:

Multiple modern baselines: NPE, BayesFlow, and Flow Matching
Standard benchmark problems: MoG Benchmark, SBI benchmark (SLCP (with and w/o distractors), 2-moons), image-based problem
Established evaluation metrics: C2ST, simulation budget, runtime
Regarding soundness:

R: "The intent of this paper, its methodology, and the numerical experiments it claims to have conducted are not clear."

A: We would welcome the reviewer identifying which specific results (numerical experimentes) are not clear, to address them. The score indicates "Major errors (e.g., an incorrect theorem or derivation)" but the reviewer provides no reference to equation numbers, theorem references, etc. that considers as incorrect. Which specific theorem or derivation contains an error?

Significance and novelty:

R: The reviewer claims we present "known definitions and results...as part of something new" but provides no specific examples.

A: We would welcome the reviewer identifying which specific results they consider known, but then stated to be something new?

Unfortunately, without concrete feedback, references to line numbers, equations or specific criticisms, we cannot meaningfully respond.

We respectfully request that the reviewer provide:

Specific examples of the claimed major errors
Citations for which results are "known"
Concrete suggestions for improving clarity
Follow-up on our rebuttal
Official Commentby Authors (Christos Diou, Vasileios Gkolemis, Michael U. Gutmann)06 Dec 2025, 20:44 (modified: 06 Dec 2025, 20:45)Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewers SubmittedRevisions
Comment:
Dear Reviewer 4f6x,

Thank you again for your constructive review. We would like to kindly check whether any points remain unclear or would benefit from further clarification. Since the discussion period ends in about three days, we would greatly appreciate any feedback you might have, so that we can address any remaining concerns in time.

Kind regards,

The Authors

 Replying to Follow-up on our rebuttal
comment after rebuttal
Official Commentby Reviewer 4f6x08 Dec 2025, 18:31Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewers Submitted
Comment:
Thank you for providing your response to my review. After looking through the other reviewer's comments and also looking into the broader ROMC papers your cited, I conclude that I did not understand the intent of the methods and the merit of the approaches.

That said, I did not get enough time to revise my assessment since the timeline for this rebuttal process also coincided with NeurIPS. I do not see a way to withdraw my review, so I will mark down my expertise in making an assessment to the point where the AE can effectively ignore my assessment. I'll leave a note to that effect to them too.

 Replying to comment after rebuttal
Response to reviewer 4f6x
Official Commentby Authors (Christos Diou, Vasileios Gkolemis, Michael U. Gutmann)10 Dec 2025, 10:47Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewers Submitted
Comment:
Thank you for your openness and for taking the time to revisit the context. We appreciate your note to the AE.

Official Review of Submission1393 by Reviewer Ajx1
Official Reviewby Reviewer Ajx110 Nov 2025, 14:27 (modified: 13 Dec 2025, 00:33)Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer Ajx1Revisions
Summary And Contributions:
The authors are proposing extensions to the Robust Optimization Monte Carlo framework (ROMC) for simulation-based inference. The authors propose to include multiple observations by means of a factorized likelihood where each factor takes the form of a sample-average approximation of the probability of being 
-close to a corresponding observation. Correspondingly, they propose a uniform mixture proposal distribution (with one component per observation and seed), which they optimize component-wise to minimize the distance to each observation. The authors further propose to use a sensitivity based masking strategy to exclude uninformative dimension from the distance computation.

Paper Keywords: simulation-based inference, importance sampling, variational methods
Expertise Keywords: approximate inference, importance sampling, variational methods
Soundness: Correct / minor errors (e.g., typo or errors that do not affect the main results)
Soundness Justification:
To the best of my knowledge the methodology and presented results are sound.

Significance: Somewhat significant (e.g., theoretical contribution of limited novelty or empirical gains missing baselines/statistical rigor).
Significance Justification:
With small ε and well trained proposals the approximate posterior samples will be concentrated in a narrow region around the observed data. While this is likely problematic in high dimensional settings with many well separated observations, it might also lead to a deceptively good C2ST score in low dimensional settings with few (or tightly clustered) observations. Unfortunately the authors do not report coverage metrics or perform posterior predictive checks (for the same parameters but with new observations). As such it is hard to tell if the choice of small epsilon together with the tightly fitted proposal distributions is problematic. For SLCP the authors provide a visualization (Figure , very right) for qualitative evaluation of samples considering the joint set of observations. However, this is a low dimensional setting with only 4 observations (corresponding to the 4 left-most figures) and it only shows a subset of high importance weight samples. In this context it would be useful to report the overall number of samples and effective sample size of the weight to get a sense of how many draws it takes to cover the posterior distribution sufficiently well. Unfortunately, for the high dimensional image setting there are no baselines and no C2ST scores.

Novelty: New results
Novelty Justification:
To the best of my knowledge the presented extensions to ROMC are novel.

Non-conventional Contributions:
N/A

Clarity:
The paper is well written and the methodology and relevant concepts are clearly explained. The authors explicitly discuss the limitations of their methodology, which is great, but unfortunately they do not quantify how severe these limitations really are in their experiments.

Relation To Prior Work: All related works are clearly discussed
Additional Comments:
The labels and legends of some figures are very small.

Reproducibility: Sufficient amount of details available for reproducing the main results
Rating: 6: Accept (Technically solid paper, with high impact on at least one sub-area. The contribution is convincing and the evaluation is adequate, though there may be some limitations or open questions.)
Confidence: 3: Fairly confident (It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.)
Rebuttal by Authors
Rebuttalby Authors (Christos Diou, Vasileios Gkolemis, Michael U. Gutmann)27 Nov 2025, 21:07 (modified: 01 Dec 2025, 15:15)Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewers SubmittedRevisions
Rebuttal:
We sincerely thank the reviewer for the insightful comments, especially the ones about posterior coverage, ESS and the high-dimensional image experiment. We have performed additional analyses to fully address the raised points, where possible.

Significance Justification:

R: Concern over small 
 -- "With small ε ... is problematic."

A: The reviewer's concern is that a small 
 may lead to a restricted posterior spread. We acknowledge this concern but clarify that the spread is primarily driven by simulator noise (see point 1 below) and the multiple seed stratagy (see point 2 below), not solely by 
 threshold.

To clarify:

 constrains in data space, not parameter space.

The threshold 
 acts on the distance between simulated data points 
, not the distance in the parameter space 
). A small 
 can still yield a broad region if the simulator exhibits low sensivity to changes around 
. Therefore, although it is true that decreasing 
 will decrease the (per-seed) proposal region, this is not done in a linear manner.

Muliple seeds naturally increase coverage.

For each observation we run 
 seeds, each producing its own proposal region. Even if each seed-specific region is individually tight, simulator randomness ensures they to land in different parts of the parameter space. Their union naturally expands the overall posterior support.

Furthermore, the reviewer correctly notices the absence of (global or local) coverage diagnostics; unfortunately, these are prohibitively expensive for non-amortized methods like R2OMC as they require generating a large calibration set of prior samples and associated simulations and performing inference for every calibration point (Deistler et al., Simulation-Based Inference: A Practical Guide).

However, we have performed Posterior Predictive Checks (PPCs). These checks confirm that the distribution of simulated data generated from the approximate posterior is consistent with the observed data, demonstrating that our posterior does not exhibit major failure modes. Specifically, in PPCs, for a given observation(s), we draw 
 posterior samples and simulate one output for each. We then verify that the observation is consistent with distribution of these simulated outputs. A well-calibrated posterior should reproduce the observation within simulator noise (as is the case in SLCP). These PPC results will be included in our camera-ready Appendix.

R: "For SLCP the autho ... no C2ST scores."

A: We have calculated the following efficiency statistics:

Overall Efficiency: Our approach yields an overall efficiency of approximately 
 per cent. This means we generate about 
 proposal samples for every 
 final retained posterior sample.

Sample Generation Breakdown:

In a typical run, we generate 
 total proposals (4 observations by 5,000 each)
Approximately 4,000 (
 per cent) are initially accepted (receive a positive weight), and, finally, we resample from this weighted pool to retain 1,000 final posterior samples.
We then resample from this weighted pool to retain 1,000 final posterior samples.
Effective Sample Size (ESS): The ESS depends on the particular run and simulation budget. We have performed some experiments that show that, on a typical run, the ESS of the weighted pool of accepted samples is 
, which confirms that the weights are not degenerate and the pool contains sufficient independent information to support the target size of posterior draws.

We will include precise statistics for the above metrics in the revised manuscript.

R: "Unfortunately, for the high dimensional image setting there are no baselines and no C2ST scores."

A: Thank you for your valuable feedback regarding the need for quantitative metrics. We agree that including appropriate metrics is essential for a thorough evaluation.

Baselines:

We acknowledge the absence of common baselines (like NPE or BayesFlow) for the high-dimensional image setting. Our initial exploratory experiments indicated that these methods struggle significantly with raw high-dimensional outputs, likely requiring extensive pre-processing, dimensionality reduction, or specialized architectures to perform competitively. Working on these modifications is beyond the scope of the current work. We therefore opted to focus on directly demonstrating R2OMC's capability in effectively handling high-dimensional data under limited simulation budgets, rather than presenting potentially misleadingly poor baseline performance.

C2ST: We apologize for the omission of the C2ST scores for the first version of the high-dimensional image experiment. As the simulator applies an affine transformation + noise, computing the ground truth posterior is indeed feasible. We have now computed these scores using 1000 generated samples (from the ground truth and the approximate posterior) across 10 distinct observations. The resulting C2ST score is 
. We will ensure this metric is included in the revised manuscript. For the second version, it is not so straightforward to obtain ground truth posterior due the checkerboard filter.

R: "The paper is well written ... in their experiments."

A: We thank the reviewer for the positive feedback on the clarity of the paper and our discussion of its limitations. In our experiments, these limitations did not materially affect R2OMC's performance.

Specifically:

Limitation 1 -- Requirement for differentiable simulators: All simulators used in our experiments were intentionally chosen to be differentiable, so this constraint did not hinder performance.

Limitation 2 -- Performance depends on successful optimization: Gradient-based optimization was generally successful in our tests, and we did not observe major issues. Nevertheless, local minima or non-convergence can occur and would negatively affect R2OMC' performance. We therefore recommend that practitioners always inspect the distances at the optimization endpoints: if many 
 values do not approach zero, this is a clear signal that optimization issues need to be addressed.

Limitation 3 -- ε may grow large to cover all observations: In SLCP, 
 remained well-controlled. Observation-based proposal regions (per seed and observation) were typically built with ε≈0.5 and samples drawn from them were tested to "match" with all observations. The threshold to retain a positive weight was set to about ε≈2.0. If the distance with respect to each observation (at least one seed-specific distance per observation) was below that threshold, then they were weighted following Eq. 14. So, 
 practical and stable in our runs.

Limitation 4 -- Optimization does not explicitly incorporate the prior.: This can push proposal regions into low- or zero-prior areas. In the 2-moons simulator, for example, we observe that some endpoints 
 fell outside the prior support. However, because a sufficient number of endpoints remained within the prior mass, inference accuracy was preserved, and overall performance was not meaningfully affected. Note that, when this limitation does not harm performance, it can be viewed as a feature: we can easily change the prior without expensive further compute and thus easily perform prior sensitivity analysis.

R: "The labels and legends of some figures are very small."

A: We have noticed that and we will fix it.

Follow-up on our rebuttal
Official Commentby Authors (Christos Diou, Vasileios Gkolemis, Michael U. Gutmann)06 Dec 2025, 20:42 (modified: 06 Dec 2025, 20:45)Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewers SubmittedRevisions
Comment:
Dear Reviewer Ajx1,

Thank you again for your constructive review. We would like to kindly check whether any points remain unclear or would benefit from further clarification. Since the discussion period ends in about three days, we would greatly appreciate any feedback you might have, so that we can address any remaining concerns in time.

Kind regards,

The Authors

 Replying to Rebuttal by Authors
Re: Rebuttal by Authors
Official Commentby Reviewer Ajx109 Dec 2025, 10:56Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewers Submitted
Comment:
Thanks for addressing my concerns and putting in the effort to run additional analysis and evaluation. In light of this rebuttal, I'll raise my score and recommend acceptance.

 Replying to Re: Rebuttal by Authors
Response to reviewer Ajx1
Official Commentby Authors (Christos Diou, Vasileios Gkolemis, Michael U. Gutmann)10 Dec 2025, 10:49Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewers Submitted
Comment:
Thank you for appreciating our added work which led to your positive recommendation.

Official Review of Submission1393 by Reviewer q1TS
Official Reviewby Reviewer q1TS09 Nov 2025, 19:04 (modified: 13 Dec 2025, 00:33)Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewers Submitted, Reviewer q1TSRevisions
Summary And Contributions:
The paper presents an efficient simulation-based inference (SBI) approach for parameter inference problems involving differentiable simulators. The method is based on Optimization Monte-Carlo framework, and leverages gradient information to drive the inference search towards posterior regions of high-density. The proposed R2OMC approach is shown to be effective and efficient in comparison to existing methods. The authors also demonstrate robustness against distractors, and the ability to perform parameter inference using multiple observations. Therefore, the approach is a welcome addition to modern SBI literature. However, my overall rating has been influenced by two things which the authors may consider (see the additional comments section for more details):

The related works can be expanded further, and
The authors can consider an additional baseline (such as Simformer), or at least discuss/motivate the exclusion of some more recent SBI approaches.
Paper Keywords: simulation-based inference, optimization Monte-Carlo, Bayesian inference
Expertise Keywords: Bayesian inference, optimization, probabilistic machine learning
Soundness: Correct / minor errors (e.g., typo or errors that do not affect the main results)
Soundness Justification:
The approach builds upon optimization Monte-Carlo, and has been explained well in terms of theory. The experiments are well-designed and conclusively support the claims.

Significance: Significant contributions (e.g., strong theoretical insights or well-supported empirical improvements with appropriate baselines/statistics).
Significance Justification:
The paper presents a fresh perspective to SBI for differentiable simulators, from an optimization lens. The gains offered by the approach in terms of efficiency are significant, and the approach also performs well in presence of distractors and is able to leverage multiple observations to infer from.

Novelty: New results
Novelty Justification:
There has been a lot of activity within SBI, based on generative modeling approaches. This R2OMC approach here approaches SBI from an optimization perspective, minimizing the distance between observed data and simulated responses. Since differentiable simulators are considered, the approach leverages gradient information to iteratively and quickly refine the posterior (and also avoid the effect of distractors). Therefore, the approach brings novelty in its construction and adaptation to SBI.

Non-conventional Contributions:
To a limited extent - since a vast majority of modern SBI approaches use some form of generative modeling, while R2OMC is optimization-based.

Clarity:
The paper is well written, with neatly organized sections and experimental design. I have minor concerns with respect to readability in Figures, which I mention in additional comments below. However, the related works section can be made more comprehensive (comments below).

Relation To Prior Work: Some minor related works are missing or described incorrectly
Additional Comments:
I mention minor typographical errors and some questions/suggestions below.

Why has Simformer [1] not been considered as a baseline? It is a very accurate approach, but quite expensive. I mention this also because R2OMC might potentially match Simformer, at substantially lower computational cost. The absence of this baseline has not factored in my review, since I feel the point of the proposed approach is efficiency, and there R2OMC already holds an advantage by design, but the paper might benefit from including this comparison (especially since you mention systematic comparison to 'state-of-the-art' SBI methods (end of Sec 1), which Simformer is part of in my view).
Another interesting paper to cite in related work is the BOLFI framework [2]. They use Bayesian optimization to pose SBI as a minimization problem, similar in spirit in some ways.
Fig. 2 appears a bit blurry (if I were nitpicking).
Sec 3.3, typo: Algorithms 1 -> Algorithm 1.
Fig. 5 has some text (Type: inferred/ground truth), which is very hard to read. Perhaps it can be moved into the caption to enhance readability.
Fig. 6 (same as above).
The related works section wrt more recent approaches can be expanded further. See related works in [3] for references.
[1] Gloeckler, M., Deistler, M., Weilbach, C.D., Wood, F. and Macke, J.H., 2024, July. All-in-one simulation-based inference. In International Conference on Machine Learning (pp. 15735-15766). PMLR.

[2] Gutmann, M.U. and Corander, J., 2016. Bayesian optimization for likelihood-free inference of simulator-based statistical models. Journal of Machine Learning Research, 17(125), pp.1-47.

[3] Vetter, J., Gloeckler, M., Gedon, D. and Macke, J.H., 2025. Effortless, Simulation-Efficient Bayesian Inference using Tabular Foundation Models. arXiv e-prints, pp.arXiv-2504.

Reproducibility: Sufficient amount of details available for reproducing the main results
Rating: 5: Borderline Accept (Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation.)
Confidence: 4: Confident, but not absolutely certain. (It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.)
Rebuttal by Authors
Rebuttalby Authors (Christos Diou, Vasileios Gkolemis, Michael U. Gutmann)28 Nov 2025, 11:51 (modified: 01 Dec 2025, 15:15)Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewers SubmittedRevisions
Rebuttal:
Additional Comments:

R: "Why has Simformer [1] not been considered as a baseline?... in my view."

A: Thank you for this suggestion. Simformer is indeed an important recent SBI method and we will include a discussion of it in the Related Work (Background) section.

Simformer offers several strong advantages: it can sample arbitrary conditionals of the joint distribution, handle missing or unstructured data, support function-valued parameters, and incorporate known dependency structure to improve accuracy. Our main reason for not including it as a baseline is its computational cost. Simformer, like most neural generative models, typically requires on the order of 
 simulations (with occasional reports of 
) to reach peak performance. In contrast, R2OMC is explicitly designed for efficiency and usually operates with only hundreds to a few thousand simulations by directly leveraging simulator gradients. For this reason, Simformer would not affect the central message of the paper, which focuses on efficiency.

Nevertheless, we agree that Simformer is a highly relevant accuracy-oriented reference. We will add a citation and discussion, and, if computationally feasible, consider including a quantitative comparison in the appendix of the final version.

R: "Another interesting paper to cite in related work is the BOLFI framework [2]...similar in spirit in some ways."

A: Thank you for this suggestion. BOLFI is indeed an important method with conceptual similarities to our approach, as it also tackles inference through optimization. Due to that, it significantly reduces the required simulations compared to naive ABC methods (see Section 7 of the paper). However, as it relies on Bayesian Optimization and considers the underlying simulator as black-box (does not use simulator's gradients directly), it does not scale well to higher-dimensional problems . As a result, it is tested against low-dimensional settings (e.g., the three-parameter Ricker model). For this reason, we did not include BOLFI in our systematic evaluation, but we agree that its conceptual connection is important and will add the relevant discussion.

R: "Fig. 2 appears a bit blurry (if I were nitpicking) ... Fig. 6 (same as above)."

A: We apologize for the readability issues and thank the reviewer for pointing them out. We will re-render all relevant figures (specifically Figs. 2, 5, and 6) as high-resolution vector graphics. For Figs. 5 and 6, we will remove redundant text (inferred ground/truth) and include it only once in the rightmost panel to improve readability.

R: "The related works section wrt more recent approaches can be expanded further. See related works in [3] for references."

A: We agree that expanding the Related Work section will strengthen the paper. We will incorporate a broader discussion of recent SBI advances, and we thank the reviewer for pointing us to [3]; we will use it to enrich the coverage of recent literature.

 Replying to Rebuttal by Authors
Official Comment by Reviewer q1TS
Official Commentby Reviewer q1TS06 Dec 2025, 01:48Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewers Submitted
Comment:
I thank the authors for the rebuttal. Indeed, I did not mean to suggest BOLFI as a competing approach, rather an interesting and related approach to discuss in the related works section. I appreciate the responses and welcome the suggested changes proposed by the authors.

 Replying to Official Comment by Reviewer q1TS
Response to reviewer q1TS
Official Commentby Authors (Christos Diou, Vasileios Gkolemis, Michael U. Gutmann)10 Dec 2025, 10:50Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewers Submitted
Comment:
Thank you for your constructive feedback. We appreciate your suggestions and will incorporate the proposed changes.

 Replying to Response to reviewer q1TS
Official Comment by Reviewer q1TS
Official Commentby Reviewer q1TS11 Dec 2025, 18:04Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewers Submitted
Comment:
Thanks, I also appreciate the new Lotka-Volterra results - this completes the paper further, and I am happy to raise my rating to Accept (6) based on the proposed changes during the rebuttal. (I will do so when the option becomes available to me in Openreview).
