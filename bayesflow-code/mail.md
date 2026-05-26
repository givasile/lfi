Hi guys,

Michael and I are currently attending the Approximately Bayes seminar and we talked about your AISTATS paper again (congrats on the acceptance!)

As discussed, I believe the results presented in Figure 3 are not representative of the SOTA in the field (even if it's what one gets when one uses the SBI package) and do not do full justice to our work on the bayesflow software.

Attached is a fully working notebook using the recent bayesflow version that gets you a C2ST below 0.75 with 18 distractors (the hardest setting in your paper) for 11 (!) seconds on 1/5 of the training budget used in the paper.

I took a deeper dive into the results code (https://github.com/givasile/lfi/blob/aistats2026/code/scripts/mog_benchmark/two_modes_distractors.py) and don't think that the settings used were comparable between the models or optimal (even in defense of the sbi package). For example, using 32 layers for a summary network is an overkill for a problem that doesn't need a summary network (btw, the sbi implementation is definitely not my implementation of the old version of the coupling flow + summary).

Additionally, I ran a case with D = 1000 distractors to push the limits of what bayesflow can do for a fixed budget and default settings (no fancy gating or variable selection layers or input dropout). I can still get a C2ST of around 0.75 for around a minute but you can clearly see that the network overfits for the fixed budget of 100k.

Just to be clear, I am absolutely convinced that Optimization Monte Carlo is super cool and even suggested thinking of a collab for optimal design + posterior inference (https://arxiv.org/pdf/2512.22999). What I am afraid of is that looking at Figure 3, ppl can be lead to believe that all of these methods are junk for something as trivial as a GMM, whereas the attached notebook shows that they start to get junkish for a 50x increase in distractors and 1/60 of the runtime in your experiment. With some tuning and a proper first layer (e.g., PCA), I can easily go up to D = 100k distractors, so even this would hold for a completely naive "default" setting.

One more conceptual thing: to me, both flow matching and the fake bayesflow in your paper are NPE. We view NPE as an approach, not an architecture. BayesFlow has been, as of 2023, a meta-framework, e.g., you can do NPE with flow matching, diffusion, normalizing flows, etc with any deep learning backend. So perhaps these things can be clarified in a post publication note (we did the same for our recent iclr paper where we messed up a 1/n factor we carelessly took from another paper...)

Cheers and hope to stay in touch,
Stefan
