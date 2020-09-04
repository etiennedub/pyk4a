SOURCES=pyk4a example tests

.PHONY: setup fmt lint test help build
.SILENT: help
help:
	echo  \
		"Available targets: \n" \
		"- setup: Install required for development packages\n" \
		"- build: Build and install pyk4a package\n" \
		"- fmt: Format all code\n" \
		"- lint: Lint code syntax and formatting\n" \
		"- test: Run tests\n"\
		"- test-hardware: Run tests related from connected kinect"
		"- test-no-hardware: Run tests without connected kinect"


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
	pytest --cov=pyk4a --verbose

test-hardware:
	pytest --cov=pyk4a  -m "device" --verbose

test-no-hardware:
	pytest --cov=pyk4a  -m "not device" --verbose