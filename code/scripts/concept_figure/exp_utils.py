import os
import time
import numpy as np
import matplotlib.pyplot as plt
import lfi
import torch
# ----------------------------- #

def run_inference(
        prior, simulator, observation, samples_gt,
        method_name,
        inference_class,
        budget,
        nof_samples,
        fit_kwargs,
        sample_kwargs,
        seed=42,
):
    """Run inference, sample, plot, and save C2ST."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    inference = inference_class(
        prior=prior,
        simulator=simulator,
        observation=observation,
    )

    tic = time.time()
    inference.fit(
        budget=budget,
        fit_kwargs=fit_kwargs or {}
    )

    samples = inference.sample(
        nof_samples=nof_samples,
        sample_kwargs=sample_kwargs or {}
    )
    runtime = time.time() - tic
    c2st_score = lfi.evaluation.c2st(samples, samples_gt)
    print(f"[{method_name}] C2ST: {c2st_score:.4f}, Runtime: {runtime:.2f} seconds")
    return c2st_score, runtime, samples

def plot_samples(figure_path, samples, gt_samples, method_name, title=None, runtime=None, c2st=None):
    """Plot posterior samples and save figure."""
    plt.figure(figsize=(8, 6))
    plt.gca().set_facecolor('#0d1b2a')
    if gt_samples is not None:
        plt.scatter(gt_samples[:, 0], gt_samples[:, 1], alpha=0.2, color='red')
    if samples is not None:
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.2, color='white')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    if title is not None:
        plt.title(title)

    if runtime is not None:
        # place some text in the bottom right corner at white background
        # make sure the text is readable, i.e, covers 1/8 of the figure
        plt.text(
            0.95,
            0.05,
            f"{runtime:.2f}s",
            transform=plt.gca().transAxes,
            fontsize=45, color='black',
            horizontalalignment='right', verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=0.95, edgecolor='none')
        )
    if c2st is not None:
        # place some text in the top left corner at white background
        plt.text(
            0.05,
            0.95,
            f"C2ST: {c2st:.2f}",
            transform=plt.gca().transAxes,
            fontsize=45, color='black',
            horizontalalignment='left', verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.95, edgecolor='none')
        )
    plt.savefig(os.path.join(figure_path, f"posterior.png"), bbox_inches='tight')
    plt.savefig(os.path.join(figure_path, f"posterior.pdf"), bbox_inches='tight')
    # plt.show(block=False)


def save_stats(dir_path, samples, gt_samples, method_name, title, c2st_score, runtime, seed=42):
    # dir path append method name
    dir_path = os.path.join(dir_path, method_name)
    os.makedirs(dir_path, exist_ok=True)

    # inside dir path check for previous runs (named as run_0, run_1, ...)
    # if not exists create run_0, else create run_{n+1}
    run_id = 0
    while os.path.exists(os.path.join(dir_path, f"run_{run_id}")):
        run_id += 1
    dir_path = os.path.join(dir_path, f"run_{run_id}")
    os.makedirs(dir_path, exist_ok=True)

    # store seed
    out_file = os.path.join(dir_path, f"seed.csv")
    np.savetxt(out_file, [seed], delimiter=",")

    # store samples
    out_file = os.path.join(dir_path, f"samples.csv")
    np.savetxt(out_file, samples, delimiter=",")

    # store C2ST score
    out_file = os.path.join(dir_path, f"c2st.csv")
    np.savetxt(out_file, [c2st_score], delimiter=",")

    # store runtime
    out_file = os.path.join(dir_path, f"runtime.csv")
    np.savetxt(out_file, [runtime], delimiter=",")

    # plot samples
    plot_samples(dir_path, samples, gt_samples, method_name, title, runtime, c2st_score)
