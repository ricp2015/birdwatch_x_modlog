#################################################################################
# GLOBALS                                                                       #
#################################################################################

PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt


## Show the reproducible Typer pipeline
.PHONY: pipeline-help
pipeline-help:
	$(PYTHON_INTERPRETER) -m src.pipeline --help


## Run all repeatable model evaluations
.PHONY: methods
methods:
	$(PYTHON_INTERPRETER) -m src.pipeline run methods


## Prepare the three materialized K-fold collections
.PHONY: kfold
kfold:
	$(PYTHON_INTERPRETER) -m src.pipeline prepare kfold


## Run all methods on the materialized K-fold collections
.PHONY: methods-kfold
methods-kfold:
	$(PYTHON_INTERPRETER) -m src.pipeline run methods --k-fold


## Regenerate all comparison tables and graphs
.PHONY: graphs
graphs:
	$(PYTHON_INTERPRETER) -m src.pipeline run graphs


## Generate K-fold comparison graphs (mean and standard deviation)
.PHONY: graphs-kfold
graphs-kfold:
	$(PYTHON_INTERPRETER) -m src.pipeline run kfold-graphs
	



## Delete all compiled Python files
.PHONY: clean
clean:
	$(PYTHON_INTERPRETER) -c "from pathlib import Path; [path.unlink() for path in Path('.').rglob('*.py[co]')]"
	$(PYTHON_INTERPRETER) -c "from pathlib import Path; import shutil; [shutil.rmtree(path) for path in Path('.').rglob('__pycache__')]"


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	$(PYTHON_INTERPRETER) -m venv .venv
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



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
