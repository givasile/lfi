# Code — R2OMC Experiments

This directory contains all code to reproduce the experiments in the paper
"Fast and Robust Simulation-Based Inference With Optimization Monte Carlo".

## Structure

```
lfi/          - Core Python package (inference methods, simulators, priors, evaluation)
scripts/      - Experiment scripts, one subdirectory per benchmark
results/      - Stored experiment outputs (CSVs, figures)
requirements.txt
```

## Setup

```bash
conda create -n lfi python=3.10
conda activate lfi
pip install -r requirements.txt
pip install -e .
```

## Reproducing Experiments

### MoG Benchmark (Section 4.1)
```bash
python scripts/mog_benchmark/single_mode.py
python scripts/mog_benchmark/single_mode_distractors.py
python scripts/mog_benchmark/two_modes.py
python scripts/mog_benchmark/two_modes_distractors.py
python scripts/mog_benchmark/results.py   # generate figures
```

### SBIBM Benchmarks (Section 4.2)
```bash
python scripts/sbibm/slcp.py
python scripts/sbibm/slcp_distractors.py
python scripts/sbibm/two_moons.py
python scripts/sbibm/results.py           # generate figures
```

### Lotka-Volterra (Section 4.3)
```bash
python scripts/lotka_volterra/lotka_volterra.py
python scripts/lotka_volterra/results.py  # generate figures
```

### Concept Figure
```bash
python scripts/concept_figure/simple.py
python scripts/concept_figure/high_dim.py
python scripts/concept_figure/distractors.py
python scripts/concept_figure/results.py
```

## Using `lfi` and R2OMC

For general use of the `lfi` package and R2OMC outside the paper experiments,
see the [master branch](https://github.com/givasile/lfi).
