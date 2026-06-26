.PHONY: setup data train infer eval lint fmt test clean

PYTHON ?= python
STAGE  ?= sr
CKPT   ?= outputs/best.ckpt
CONFIG ?= configs/default.yaml

setup:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt && pip install -e .

data:
	$(PYTHON) -m ircolor.data.prepare --config $(CONFIG)

train:
	$(PYTHON) -m ircolor.training.train --config $(CONFIG) stage=$(STAGE)

infer:
	$(PYTHON) -m ircolor.inference.predict --ckpt $(CKPT) --input $(INPUT)

eval:
	$(PYTHON) -m ircolor.eval.evaluate --ckpt $(CKPT) --config $(CONFIG)

lint:
	ruff check src tests && mypy src

fmt:
	ruff check --fix src tests && black src tests

test:
	pytest

clean:
	rm -rf outputs/*.ckpt .pytest_cache **/__pycache__
