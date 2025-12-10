# Reviewer 4

Official Review of Submission1393 by Reviewer q1TS
Official Reviewby Reviewer q1TS09 Nov 2025, 19:04 (modified: 21 Nov 2025, 22:21)Program Chairs, Authors, Senior Area Chairs, Area Chairs, Reviewer q1TSRevisions

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

# Answer to Reviewer 4

**Additional Comments:**

R: "Why has Simformer [1] not been considered as a baseline?... in my view."

A: Thank you for this suggestion. Simformer is indeed an important recent SBI method and we will include a discussion of it in the Related Work (Background) section.

Simformer offers several strong advantages: it can sample arbitrary conditionals of the joint distribution, handle missing or unstructured data, support function-valued parameters, and incorporate known dependency structure to improve accuracy. Our main reason for not including it as a baseline is its computational cost. Simformer, like most neural generative models, typically requires on the order of $10^5$ simulations (with occasional reports of $\approx 10^4$) to reach peak performance. In contrast, R2OMC is explicitly designed for efficiency and usually operates with only hundreds to a few thousand simulations by directly leveraging simulator gradients. For this reason, Simformer would not affect the central message of the paper, which focuses on efficiency.

Nevertheless, we agree that Simformer is a highly relevant accuracy-oriented reference. We will add a citation and discussion, and, if computationally feasible, consider including a quantitative comparison in the appendix of the final version.

---

R: "Another interesting paper to cite in related work is the BOLFI framework [2]...similar in spirit in some ways."

A: Thank you for this suggestion. BOLFI is indeed an important method with conceptual similarities to our approach, as it also tackles inference through optimization. Due to that, it significantly reduces the required simulations compared to naive ABC methods (see Section 7 of the paper). However, as it relies on Bayesian Optimization and considers the underlying simulator as black-box (does not use simulator's gradients directly), it does not scale well to higher-dimensional problems . As a result, it is tested against low-dimensional settings (e.g., the three-parameter Ricker model). For this reason, we did not include BOLFI in our systematic evaluation, but we agree that its conceptual connection is important and will add the relevant discussion.

---

R: "Fig. 2 appears a bit blurry (if I were nitpicking) ... Fig. 6 (same as above)."

A: We apologize for the readability issues and thank the reviewer for pointing them out. We will re-render all relevant figures (specifically Figs. 2, 5, and 6) as high-resolution vector graphics. For Figs. 5 and 6, we will remove redundant text (inferred ground/truth) and include it only once in the rightmost panel to improve readability.

---

R: "The related works section wrt more recent approaches can be expanded further. See related works in [3] for references."

A: We agree that expanding the Related Work section will strengthen the paper. We will incorporate a broader discussion of recent SBI advances, and we thank the reviewer for pointing us to [3]; we will use it to enrich the coverage of recent literature.
