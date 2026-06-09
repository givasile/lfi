"""Live status for the systematic real-BayesFlow sweep.

Reads results from disk, prints two D x budget matrices (dist=18 and dist=0)
with mean +/- std C2ST per cell, and estimates remaining time + ETA based on
measured per-step training cost and the ascending/early-stop budget logic.
"""
import os, glob
import numpy as np
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "../code/results/mog_benchmark")
LOG = os.path.join(HERE, "run_systematic.log")

D_LIST = [2, 5, 10, 15, 20]
BUDGETS = [5000, 10000, 20000, 50000]
SEEDS = [42, 48930, 1234]
DISTS = [("two_modes_distractors", 18), ("two_modes", 0)]   # run order
THR = 0.8
EPOCHS, BATCH = 120, 256
OVERHEAD_FUDGE = 1.15   # sampling + c2st eval on top of training time
FALLBACK_S_PER_STEP = 0.010   # jax-CPU (measured ~10 ms/step); self-calibrates once cells land


def steps(b):
    return (b / BATCH) * EPOCHS


COMPARE_BUDGETS = [1000, 5000, 10000, 20000, 50000, 100000]


def cell_dir(sub, D, b):
    return os.path.join(BASE, sub, f"D_{D}", f"real_bayesflow_{b}")


def load_baseline():
    """Paper's labeled bayes_flow baseline: bayes_flow_<b>/run_*/c2st.csv."""
    base = {}
    for sub, dist in DISTS:
        for D in D_LIST:
            for b in COMPARE_BUDGETS:
                d = os.path.join(BASE, sub, f"D_{D}", f"bayes_flow_{b}")
                sc = []
                for rd in glob.glob(os.path.join(d, "run_*")):
                    cp = os.path.join(rd, "c2st.csv")
                    if os.path.exists(cp):
                        sc.append(float(np.loadtxt(cp)))
                base[(dist, D, b)] = sc
    return base


def fmt_c(sc):
    if not sc:
        return "   --   "
    m = np.mean(sc)
    if len(sc) == 1:
        return f"{m:.3f}·1 "
    return f"{m:.3f}±{np.std(sc):.2f}"


def compare(data, base, dist):
    """Side-by-side: R = real_bayesflow (this run), B = paper bayes_flow."""
    lines = []
    head = "   D  m   " + "".join(f"{b//1000}k".rjust(10) + " " for b in COMPARE_BUDGETS)
    lines.append(head)
    for D in D_LIST:
        stopped = False
        rrow = f"  {D:<3} R: "
        brow = f"      B: "
        for b in COMPARE_BUDGETS:
            if b in BUDGETS:
                sc = data.get((dist, D, b), ([], []))[0]
                cell = fmt_c(sc)
                if sc and len(sc) == 3 and all(x <= THR for x in sc) and not stopped:
                    cell = "*" + cell.strip()
                    stopped = True
                elif stopped and not sc:
                    cell = "(skip) "
            else:
                cell = "  n/r   "   # budget not run by real_bayesflow
            rrow += f"{cell:>10} "
            brow += f"{fmt_c(base.get((dist, D, b), [])):>10} "
        lines.append(rrow)
        lines.append(brow)
    return "\n".join(lines)


def gather():
    data = {}
    for sub, dist in DISTS:
        for D in D_LIST:
            for b in BUDGETS:
                d = cell_dir(sub, D, b)
                sc, rt = [], []
                for s in SEEDS:
                    cp = os.path.join(d, f"seed_{s}", "c2st.csv")
                    rp = os.path.join(d, f"seed_{s}", "runtime.csv")
                    if os.path.exists(cp):
                        sc.append(float(np.loadtxt(cp)))
                        if os.path.exists(rp):
                            rt.append(float(np.loadtxt(rp)))
                data[(dist, D, b)] = (sc, rt)
    return data


def calibrate(data):
    persteps = []
    for (dist, D, b), (sc, rt) in data.items():
        for r in rt:
            persteps.append(r / steps(b))
    return float(np.median(persteps)) if persteps else FALLBACK_S_PER_STEP


def est_cell_train_s(b, s_per_step):
    return steps(b) * s_per_step * OVERHEAD_FUDGE


def fmt_cell(sc):
    if not sc:
        return "   --   "
    m = np.mean(sc)
    if len(sc) == 1:
        return f"{m:.3f}(1)"
    s = np.std(sc)
    n = len(sc)
    tag = "" if n == 3 else f"[{n}]"
    return f"{m:.3f}±{s:.3f}{tag}"


def matrix(data, dist):
    lines = []
    header = "  D\\budget " + "".join(f"{b//1000:>5}k   " for b in BUDGETS)
    lines.append(header)
    for D in D_LIST:
        row = f"  D={D:<2}    "
        stopped = False
        for b in BUDGETS:
            sc, _ = data[(dist, D, b)]
            cell = fmt_cell(sc)
            # mark the rung where this D succeeded (early-stop point)
            if sc and len(sc) == 3 and all(x <= THR for x in sc) and not stopped:
                cell = "*" + cell
                stopped = True
            elif stopped and not sc:
                cell = "  (skip)"
            row += f"{cell:>9} "
        lines.append(row)
    return "\n".join(lines)


def remaining_seconds(data, s_per_step):
    rem = 0.0
    for sub, dist in DISTS:
        for D in D_LIST:
            for b in BUDGETS:
                sc, _ = data[(dist, D, b)]
                done = len(sc)
                # if an EARLIER budget for this D already succeeded, this D stopped
                earlier_success = any(
                    len(data[(dist, D, bb)][0]) == 3 and all(x <= THR for x in data[(dist, D, bb)][0])
                    for bb in BUDGETS if bb < b
                )
                if earlier_success:
                    continue  # skipped by early-stop
                rem += (len(SEEDS) - done) * est_cell_train_s(b, s_per_step)
    return rem


def backend_from_log():
    if not os.path.exists(LOG):
        return "?"
    try:
        with open(LOG) as f:
            for line in f:
                if "Using backend" in line:
                    return line.strip().split("Using backend")[-1].strip().strip("'\"")
    except Exception:
        pass
    return "?"


def runtime_summary(data):
    """Total CPU training hours + mean training seconds per budget."""
    per_budget = {b: [] for b in BUDGETS}
    total = 0.0
    for (dist, D, b), (sc, rt) in data.items():
        for r in rt:
            per_budget[b].append(r)
            total += r
    lines = [f"CPU training time tracked (sum of runtime.csv): {total/3600:.2f} CPU-h"]
    cols = "  ".join(
        f"{b//1000}k={np.mean(per_budget[b]):.0f}s(n{len(per_budget[b])})" if per_budget[b]
        else f"{b//1000}k=--"
        for b in BUDGETS
    )
    lines.append(f"  mean train time / cell:  {cols}")
    return "\n".join(lines)


def log_start_time():
    if not os.path.exists(LOG):
        return None
    try:
        with open(LOG) as f:
            for line in f:
                if "scan starting at" in line:
                    hms = line.strip().split()[-1]
                    today = datetime.now().date()
                    t = datetime.strptime(hms, "%H:%M:%S").time()
                    return datetime.combine(today, t)
    except Exception:
        return None
    return None


def main():
    data = gather()
    s_per_step = calibrate(data)
    done_seeds = sum(len(sc) for (sc, rt) in data.values())

    now = datetime.now()
    print(f"=== real-BayesFlow sweep status @ {now:%Y-%m-%d %H:%M:%S} ===")
    print(f"backend: {backend_from_log()}  |  per-step train cost (median, measured): "
          f"{s_per_step*1000:.1f} ms  | overhead x{OVERHEAD_FUDGE}")
    print(runtime_summary(data))
    print()
    for sub, dist in DISTS:
        label = f"distractors = {dist}" + (" (18, core claim)" if dist == 18 else " (none)")
        print(f"[{label}]   * = early-stop rung (all 3 seeds <= {THR})")
        print(matrix(data, dist))
        print()

    base = load_baseline()
    print("=== SIDE-BY-SIDE: R = real_bayesflow (this jax run) vs B = paper bayes_flow ===")
    print(f"    (n/r = budget not run by R; * = R early-stop rung; lower C2ST = better)")
    for sub, dist in DISTS:
        print(f"\n[distractors = {dist}]")
        print(compare(data, base, dist))
    print()

    rem = remaining_seconds(data, s_per_step)
    eta = now + timedelta(seconds=rem)
    start = log_start_time()
    elapsed = (now - start).total_seconds() if start else None
    if elapsed is not None and elapsed < 0:
        elapsed += 86400   # start parsed as "today"; correct for a date rollover
    print(f"seeds done: {done_seeds}")
    if elapsed is not None:
        print(f"elapsed: {elapsed/3600:.2f} h")
    print(f"est. remaining (upper bound, assumes unfinished D run full ladder): "
          f"{rem/3600:.2f} h")
    print(f"ETA: {eta:%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
