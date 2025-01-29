# LFI 

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

*A Python package for likelihood-free inference (LFI) methods.* 

The `LFI` package is a collection of likelihood-free inference (LFI) methods 
and providing a unified interface, despite the unique parameters and inner workings of each method.

## How It Works

### Step 1: Define the Inference Method

To define an inference method, specify the prior, simulator, and observation:

```python
method = <LFI method>(prior, simulator, observation)
```

### Step 2: Perform Inference

To perform inference, Use the `fit` method.
The only required parameter is the `budget`, which specifies the number of simulations the method can run.
Additional parameters can also be passed, depending on the method.

```python
method.fit(budget, **kwargs)
```

### Step 3: Sample from the Posterior

To generate samples from the estimated posterior, use the `sample` method.
The only required parameter is `nof_samples`, with the desired number of samples to draw. 
Like `fit`, the `sample` method accepts additional parameters specific to the method.

```python
samples = method.sample(nof_samples, **kwargs)
```

### Shortcut: Fit and Sample Together

The `fit_sample` method combines the `fit` and `sample` steps, returning both the samples and the runtime.

```python
samples, time = method.fit_sample(budget, nof_samples, fit_kwargs={}, sample_kwargs={})
```

This simple interface makes it easy to use different LFI methods under a unified framework.
All implemented methods should follow this interface, allowing for easy comparison and evaluation.

## Install 

``` shell
make conda-init ENV=dev REQUIREMENTS=requirements-dev.txt
conda activate lfi-dev
pip install --upgrade sbi
pip install future
```


## Project Organization

The project follows the cookiecutter data science project template.

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         sbi and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── sbi   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes sbi a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

