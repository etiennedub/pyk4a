SOURCES=pyk4a example

.PHONY: setup fmt lint

setup:
	pip install -r requirements-dev.txt

fmt:
	isort  $(SOURCES)
	black $(SOURCES)

lint:
	black --check $(SOURCES)
	flake8 $(SOURCES)
	mypy $(SOURCES)
