#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = lfi
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


# Conda related commands
.PHONY: conda-remove-env
conda-remove-env:
	@conda env list | grep -q "^$(PROJECT_NAME)-$(ENV) " && conda env remove --name $(PROJECT_NAME)-$(ENV) -y || echo "Environment $(PROJECT_NAME)-$(ENV) does not exist, skipping removal."

.PHONY: conda-create-env
conda-create-env:
	@conda create --name $(PROJECT_NAME)-$(ENV) python=$(PYTHON_VERSION) -y

.PHONY: conda-install-requirements
conda-install-requirements:
	@conda run -n $(PROJECT_NAME)-$(ENV) pip install --upgrade pip
	@conda run -n $(PROJECT_NAME)-$(ENV) pip install -r $(REQUIREMENTS)
	@conda run -n $(PROJECT_NAME)-$(ENV) pip install -e .

.PHONY: conda-init
conda-init: conda-remove-env conda-create-env conda-install-requirements

.PHONY: conda-update
conda-update: conda-install-requirements

# Pip related commands
.PHONY: venv-remove
venv-remove:
	rm -rf .venv-$(ENV)

.PHONY: venv-create
venv-create:
	python -m venv .venv-$(ENV)

.PHONY: venv-install-requirements
venv-install-requirements:
	source .venv-$(ENV)/bin/activate && python -m pip install --upgrade pip
	source .venv-$(ENV)/bin/activate && python -m pip install -r $(REQUIREMENTS)
	source .venv-$(ENV)/bin/activate && python -m pip install -e .

.PHONY: venv-init
venv-init: venv-remove venv-create venv-install-requirements

.PHONY: venv-update
venv-update: venv-install-requirements

## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 sbi
	isort --check --diff --profile black sbi
	black --check --config pyproject.toml sbi

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml sbi




## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) sbi/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
