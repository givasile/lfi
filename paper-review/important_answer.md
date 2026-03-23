To further address the reviewer’s concern regarding applicability to a more realistic model, we conducted an additional analysis on the Lotka-Volterra model, a standard problem in the SBI literature [1]. The Lotka-Volterra model is characterized by its use of an internal system of differential equations (DE) to generate a sequence of outputs as observations, in a time-series manner. Using an underlying system of DE is a common approach for modeling natural phenomena through a simulator but, so far, we had not tested our method against such a case.

Problem Description (Full specification details are provided in [1]):

Prior: $p(\alpha, \beta, \gamma, \delta)$

α ∼ LogNormal(−0.125, 0.5)
β ∼ LogNormal(−3, 0.5),
γ ∼ LogNormal(−0.125, 0.5),
δ ∼ LogNormal(−3, 0.5)
Simulator: $x|\theta = (x_{1,1}, x_{2,1}, \ldots , x_{1,10}, x_{2,10})$, where:

$x_{1,i} \sim LogNormal(\log(X), 0.1)$
$x_{2,i} \sim LogNormal(\log(Y), 0.1)$
X, Y are simulated from:
$$\frac{dX}{dt} = \alpha X - \beta XY, \quad \frac{dY}{dt} = -\gamma Y + \delta XY$$

Dimensionality: $\theta \in \mathbb{R}^4$, $x \in \mathbb{R}^{20}$

Fixed parameters: $T = 20$, $(X(0), Y(0)) = (30, 1)$

Implementation Note

To ensure compatibility with our JAX-based framework, we implemented the Lotka-Volterra simulator directly in JAX. Since our differential equation solver slightly varies from the Julia-based implementation used in [1], we cannot leverage the ground truth samples provided by that benchmark for a direct $\text{C2ST}$ comparison. Therefore, we validate our method through Simulation-Based Calibration (SBC), a standard approach for assessing posterior quality in simulation-based inference without ground truth samples [2].

Evaluation Metrics

We conducted Simulation-Based Calibration (SBC) diagnostics across 150 independent repetitions with 100 posterior samples each, utilizing a budget of 1,000 simulator calls per repetition. SBC assesses the statistical correctness of the approximate posterior $\hat{p}(\theta|x)$ through two primary checks [2]:

Uniformity Test (SBC $\chi^2$ and p-value)
Description: We first generate a parameter $\theta_i$ from the prior, $\theta_i \sim p(\theta)$, and a corresponding observation $x_i$ from the simulator, $x_i \sim p(x|\theta_i)$. We then sample $J$ times from the approximate posterior, $\theta^{(j)} \sim \hat{p}(\theta|x_i)$. For each parameter dimension $d$, the rank of the generating parameter $\theta_{i,d}$ is computed relative to the set of $J$ posterior samples.

Expected Outcome: When the posterior is correctly calibrated, the resulting ranks (aggregated over all $150$ replications) should be approximately uniformly distributed. The $\text{SBC } \chi^2$ statistic and its associated p-value test the null hypothesis that the observed rank distribution is uniform. A high p-value (as observed in our results) indicates correct calibration.

Empirical Coverage
Description: For each parameter dimension $d$ and a desired nominal coverage level (e.g., 50%, 90%, 95%), we compute the corresponding central credible interval from the approximate posterior samples.

Expected Outcome: The empirical coverage is the fraction of the 150 replications where the true generating parameter $\theta_{i,d}$ lies within the computed credible interval. Correct calibration requires that the empirical coverage closely matches the nominal coverage level (e.g., a 90% credible interval should contain the true parameter approximately 90% of the time).

Results - Table 1

Parameter	SBC χ²	SBC p-value	Coverage 50%	Coverage 90%	Coverage 95%
α (prey birth rate)	97.79	0.544	0.500	0.900	0.953
β (predation rate)	84.32	0.870	0.500	0.927	0.940
γ (predator death rate)	81.63	0.910	0.527	0.907	0.960
δ (reproduction efficiency)	89.71	0.760	0.480	0.907	0.987
Interpretation:

SBC uniformity tests: All p-values > 0.54 indicate no evidence of miscalibration (null hypothesis: correct calibration), with rank distributions consistent with uniformity.
Empirical coverage: Credible intervals achieve coverage matching nominal levels across all parameters (e.g., 90% intervals achieve 90.0–92.7% empirical coverage), demonstrating well-calibrated uncertainty quantification.
These results, obtained within a budget of 1,000 simulator calls, enhance the evidence of R2OMC's applicability to real-world problems which involve complex simulators with differential equations. While a direct $\text{C2ST}$ comparison against the benchmark's pre-computed ground truth is not available, the above calibration results are strong indication of an accurate posterior estimation by R2OMC.

Representative SBC rank histograms and coverage calibration plots will be included in the revised manuscript.

[1] Lueckmann, Jan-Matthis, et al. "Benchmarking simulation-based inference." International conference on artificial intelligence and statistics. PMLR, 2021. [2] Hermans, Joeri, et al. "A trust crisis in simulation-based inference? your posterior approximations can be unfaithful." arXiv preprint arXiv:2110.06581 (2021).
