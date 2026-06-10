"""Emit clean markdown tables (R = real_bayesflow, B = paper bayes_flow) for the reply."""
import numpy as np
import monitor_status as M

data = M.gather()
base = M.load_baseline()


def cell(scs):
    if not scs:
        return "·"
    m = np.mean(scs)
    if len(scs) == 1:
        return f"{m:.3f}"
    return f"{m:.3f}±{np.std(scs):.3f}"


def r_matrix(dist):
    out = ["| D | 5k | 10k | 20k | 50k |", "|---|---|---|---|---|"]
    for D in M.D_LIST:
        row = [cell(data[(dist, D, b)][0]) for b in M.BUDGETS]
        out.append(f"| {D} | " + " | ".join(row) + " |")
    return "\n".join(out)


def b_matrix(dist):
    cols = M.COMPARE_BUDGETS
    out = ["| D | " + " | ".join(f"{b//1000}k" for b in cols) + " |",
           "|---|" + "|".join("---" for _ in cols) + "|"]
    for D in M.D_LIST:
        row = [cell(base.get((dist, D, b), [])) for b in cols]
        out.append(f"| {D} | " + " | ".join(row) + " |")
    return "\n".join(out)


def solve_R(dist, D):
    for b in M.BUDGETS:
        sc = data[(dist, D, b)][0]
        if len(sc) == 3 and all(x <= M.THR for x in sc):
            return f"{b//1000}k"
    return "—"


def solve_B(dist, D):
    for b in M.COMPARE_BUDGETS:
        sc = base.get((dist, D, b), [])
        if sc and np.mean(sc) <= M.THR:
            return f"{b//1000}k"
    return "—"


print("## R = real_bayesflow (FlowMatching, no summary net, jax, 120 epochs, 3 seeds)\n")
for sub, dist in M.DISTS:
    print(f"### distractors = {dist}")
    print(r_matrix(dist) + "\n")

print("## B = paper bayes_flow baseline (3 runs)\n")
for sub, dist in M.DISTS:
    print(f"### distractors = {dist}")
    print(b_matrix(dist) + "\n")

print("## Budget to solve (smallest budget with C2ST <= 0.8; '—' = not reached)\n")
print("| D | R (dist=18) | B (dist=18) | R (dist=0) | B (dist=0) |")
print("|---|---|---|---|---|")
for D in M.D_LIST:
    print(f"| {D} | {solve_R(18,D)} | {solve_B(18,D)} | {solve_R(0,D)} | {solve_B(0,D)} |")
