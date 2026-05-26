# TODO — new machine

## Setup (run once, from repo root)
```bash
conda create -n lfi-bayesflow python=3.10 -y
conda activate lfi-bayesflow

# Base lfi env (same recipe as code/README.md)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sbi pyabc
pip install sbibm --no-deps
pip install -r code/requirements.txt

# BayesFlow (pinned to match the env used to draft the reply)
pip install bayesflow==2.0.8

# verify
python -c "import os; os.environ['KERAS_BACKEND']='torch'; import bayesflow as bf; print(bf.__version__)"
```

---

## Process 1 — Draft the reply to Stefan (~30 min)

Run the quick scan. Results print line-by-line as they finish.

```bash
cd bayesflow-code
KERAS_BACKEND=torch python quick_scan.py | tee /tmp/quick_scan.log
```

**What it tests:** D∈{2,5,10,20} × distractors∈{0,18} × budgets∈{1k,5k,10k}, 1 seed each.

**When done:** open `reply_draft.md`, replace the provisional D=2 numbers (marked ⚠️)
with the new rows from `quick_scan.log`, then send the reply to Stefan.

---

## Process 2 — Systematic evaluation to update the paper (~hours, run overnight)

```bash
cd bayesflow-code
KERAS_BACKEND=torch python run_bayesflow_experiments.py | tee /tmp/bf_grid.log
```

**What it tests:** D∈{2,5,10,15,20} × budgets∈{5k,10k,20k,50k,100k}, 3 seeds, 50 epochs.
Results land automatically in `code/results/mog_benchmark/two_modes_distractors/D_<D>/real_bayesflow_<budget>/`.

**Check progress at any time:**
```bash
for d in D_2 D_5 D_10 D_15 D_20; do
  for budget in 5000 10000 20000 50000 100000; do
    dir="code/results/mog_benchmark/two_modes_distractors/$d/real_bayesflow_${budget}"
    seeds=$(ls "$dir" 2>/dev/null | wc -l)
    [ "$seeds" -gt 0 ] && mean=$(cat "$dir"/*/c2st.csv | awk '{s+=$1;n++} END {printf "%.3f",s/n}') && echo "$d budget=$budget seeds=$seeds mean_c2st=$mean"
  done
done
```

**When done:**
1. Check R2OMC still clearly wins at D=10/15/20 (main paper claim)
2. Regenerate figures: `cd code/scripts/mog_benchmark && python results.py`
   - adds `real_bayesflow` as a new method — just add `"real_bayesflow"` to the `methods` list in `results.py`
3. Rename old "BayesFlow" → "NPE (SBI)" in Figure 3 captions and appendix config table
4. Commit updated figures and LaTeX
