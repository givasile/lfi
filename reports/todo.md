# Experiments

1. Concept figure
2. MoG benchmark
3. SLCP + SCLP with distractors
4. 2-moons
5. Lotka-Volterra (still pending)
6. Images dataset


# TODO

finished:
- concept figure
- mog_benchmark
- multiple_observations
- sbibm: two moons

todo:
- ~~concept figure: select plots to show~~ 
- ~~mog_benchmark:~~ 
  - ~~check plots and maybe pick correct results~~
  - maybe create a second experiment, possibly the old concept figure (last priority)
- sbibm: lotka-volterra
- sbibm: two moons: just align the code and the plot 
- images examples: just align the code

# Reviewers' feedback

## Point 1: You miss some SotA competitors and the ones you use are possibly in their vanilla version.

On v.1 we compared against NPE, NLE, Naive-ROMC.
I did some research and followed reviewers' suggestions, so in v.2 I compare against:

- NPE: same as in v.1 but with some tuning (NSF as backbone architecture)
- SNPE: sequential version of NPE
- BayesFlow: https://arxiv.org/pdf/2003.06281, in practice it is NPE with a "learnable" embedding net on top
- FMPE: aka FlowMatching https://proceedings.neurips.cc/paper_files/paper/2023/file/3663ae53ec078860bb0b9c6606e092a0-Paper-Conference.pdf; has gained a lot of attention recently

Our key take-away is confirmed by the additional experiments; with increased dimensionality or distractors, neural-based
approaches require a very high (100K) simulation budget to succeed and, in some cases, they fail even at that budget.
In contrast, we are successful with <1K simulations which the above methods cannot approach.

I have already updated the concept figure to include the above results. I have to do the same for MoG framework plots (Section 4.1).

I have not compared against GATSBY (https://arxiv.org/abs/2203.06481), which has the fame of scaling to high-dim problems, 
because I could not use the code they provide.
However, people say that GATSBY is notoriously unstable and difficult to train (I have to find the references for that),
so this is a convincing argument if a reviewer asks about it.

## Point 2: You do not have "really" high-dim and/or real world examples.

Reviewers pointed out that some extreme methods have been tested against $D > 100$.

### I have added two image examples with dimensionality = 28*28 (MNIST images).

The example is taken from [Consistency Models for Scalable and Fast Simulation-Based Inference](https://arxiv.org/pdf/2312.05440v2). 
The idea is to 
(i) select an image from the dataset (use it as $\theta^*$), 
(ii) use the simulator to generate a noisy--filter + Gaussian noise--version of the image ($x^0$), 
(iii) use R2OMC to infer the posterior and sample from it, and 
(iv) check if samples mean matches the original image ($\theta^*$).

R2OMC achieves it in low budget.
I use two distorted versions of the image:
- one with $y_i = a x_i + b  + \epsilon$ (a=0.1, b=0.5) which looks like reducing contrast
- another where I pass a checkerboard mask to the image and then add noise. 

In the latter case, the distorted image looks very different from the original image 
(sometimes it is hard to recognize the original image), 
but R2OMC still manages to recover the original image.

### Lotka-Volterra

I am close to finish it. 
If successful, it will be nice to show that R2OMC can tackle simulators with dynamics expressed as ODEs.

### SIR model: 

Not differentiable, so I have to drop it.

## Point 3: You do not deal well with Multiple Instance 

Reviewers said that typically, people use summary statistics for multiple observations, and that we do not test 
R2OMC against examples with big sets of observations.

We have to discuss that. 
I am leaning towards dropping the multiple observation part or at least not emphasizing it as a main contribution. 
I have the feeling that it fogs the main message of the paper, which is that R2OMC can handle high-dimensional problems 
with a small simulation budget, for which we provide very clear and convincing evidence.
Maybe we can keep a section on how to handle multiple observations and possibly one example, 
but not emphasize it as a main contribution.

## Some other ideas

- Add limitation 



We could remove naive-ROMC from the comparisons. 
naive-ROMC in its current version cannot handle problems of high-dimensionality
(optimization and proposal region generation becomes super slow),
so we can simply state that without further comparisons.
In v.1, we compared against a new implementation of naive-ROMC (with JAX), 
to show that it fails we have distractors or multiple observations. 
I guess this is more confusing for the reader, so we can just remove it.

## Conclusion

??

## Other comments

I’ve reimplemented everything to check for bugs and clean up the codebase.
Now I have a neat, library-like codebase compatible with all common LFI packages 
(SBI, ELFI, or JAX-based).
You have to write a simulator in three versions--JAX, PyTorch, and NumPy-- 
and then the library can perform inference using any method of the above packages.
After R2OMC, we can decide whether it’s worth go for publishing this as a library.

From September onward, we can look into XAI and LFI.
A tutorial on the topic [https://sbi-dev.github.io/sbi/latest/tutorials/07_sensitivity_analysis/](https://sbi-dev.github.io/sbi/latest/tutorials/07_sensitivity_analysis/).
Any papers I should definitely read over the summer?
