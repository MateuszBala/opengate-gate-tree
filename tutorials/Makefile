# The notebooks run in an environment of their own: they need JupyterLab and
# matplotlib, and the package needs neither. See README.md.

SHELL := /usr/bin/env bash

VENV := .venv
PYTHON := $(VENV)/bin/python
JUPYTER := $(VENV)/bin/jupyter
KERNEL := opengate-gate-tree-tutorials
VERIFY_DIR := notebooks/.tmp/verify

.PHONY: help init install kernel lab verify clean

help:
	@echo "init     create $(VENV) and install what the notebooks need"
	@echo "install  install the requirements again, into an existing $(VENV)"
	@echo "kernel   register $(VENV) with Jupyter as $(KERNEL)"
	@echo "lab      start JupyterLab in notebooks/"
	@echo "verify   run every notebook start to finish and fail on the first error"
	@echo "clean    remove $(VENV), the checkpoints and everything under .tmp/"

init:
	@command -v uv >/dev/null || { echo "uv is required: https://github.com/astral-sh/uv"; exit 1; }
	uv venv $(VENV)
	$(MAKE) install

install:
	@test -d $(VENV) || { echo "no environment yet; run: make init"; exit 1; }
	VIRTUAL_ENV=$(VENV) uv pip install -r requirements.txt

kernel:
	@test -d $(VENV) || { echo "no environment yet; run: make init"; exit 1; }
	$(PYTHON) -m ipykernel install --user --name $(KERNEL) \
		--display-name "Python (opengate-gate-tree tutorials)"

lab:
	@test -d $(VENV) || { echo "no environment yet; run: make init"; exit 1; }
	cd notebooks && ../$(JUPYTER) lab --no-browser

# Executed copies go to .tmp/, never back into the notebook: what is committed
# is what its author published. The first failure stops the run, and its
# traceback is the report.
verify:
	@test -d $(VENV) || { echo "no environment yet; run: make init"; exit 1; }
	@mkdir -p $(VERIFY_DIR)
	@set -e; for notebook in notebooks/*.ipynb; do \
		echo "running $$notebook"; \
		$(JUPYTER) nbconvert --to notebook --execute \
			--output-dir $(VERIFY_DIR) "$$notebook" >/dev/null; \
	done; \
	echo "every notebook ran"

clean:
	rm -rf $(VENV) notebooks/.tmp simulations/.tmp
	find . -name .ipynb_checkpoints -type d -prune -exec rm -rf {} +
