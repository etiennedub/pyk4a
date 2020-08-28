SOURCES=pyk4a example
TESTS=tests

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

test:
	pytest --cov=pyk4a $(TESTS)