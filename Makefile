SOURCES=pyk4a example tests
TESTS=tests

.PHONY: setup fmt lint test help
.SILENT: help
help:
	echo  \
		"Available targets: \n" \
		"- setup: Install requireded for development packages\n" \
		"- build: Build and install pyk4a package\n" \
		"- fmt: Format all code\n" \
		"- lint: Lint code syntax and formatting\n" \
		"- test: Run tests"

setup:
	pip install -r requirements-dev.txt

build:
	pip install -e .

fmt:
	isort  $(SOURCES)
	black $(SOURCES)

lint:
	black --check $(SOURCES)
	flake8 $(SOURCES)
	mypy $(SOURCES)

test:
	pytest --cov=pyk4a $(TESTS)

test-:
	pytest --cov=pyk4a $(TESTS) -m "not hardware"