PYTHON ?= uv run python
ARGS ?= "h" -n 30

.PHONY: data model train predict

data:
	$(PYTHON) data.py

model:
	$(PYTHON) model.py

train:
	$(PYTHON) train.py

predict:
	$(PYTHON) predict.py $(ARGS)
