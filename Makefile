PYTHON ?= uv run python

.PHONY: data model

data:
	$(PYTHON) data.py

model:
	$(PYTHON) model.py
