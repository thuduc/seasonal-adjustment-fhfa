# FHFA Seasonal Adjustment Pipeline Makefile

.PHONY: help install test clean docker-build docker-run lint format coverage docs

# Default target
help:
	@echo "FHFA Seasonal Adjustment Pipeline"
	@echo "================================="
	@echo "Available targets:"
	@echo "  install      Install dependencies"
	@echo "  test         Run tests"
	@echo "  coverage     Run tests with coverage report"
	@echo "  lint         Run code linting"
	@echo "  format       Format code with black"
	@echo "  clean        Clean temporary files"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run   Run pipeline in Docker"
	@echo "  docs         Generate documentation"

# Install dependencies
install:
	pip install -r requirements.txt
	cp .env.example .env

# Run tests
test:
	pytest tests/

# Run tests with coverage
coverage:
	pytest --cov=src --cov-report=html --cov-report=term

# Lint code
lint:
	flake8 src/ tests/ --max-line-length=100
	mypy src/ --ignore-missing-imports

# Format code
format:
	black src/ tests/ --line-length=100
	isort src/ tests/

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf output/*
	rm -rf .cache/*
	rm -rf .metrics/*

# Docker commands
docker-build:
	docker build -t fhfa-seasonal-adjustment:latest .

docker-run:
	docker run -v $(PWD)/data:/app/data \
	           -v $(PWD)/output:/app/output \
	           fhfa-seasonal-adjustment:latest \
	           python main.py --mode full --report

# Run pipeline locally
run:
	python main.py --mode full --report

run-adjustment:
	python main.py --mode adjustment-only

run-impact:
	python main.py --mode impact-only

# Generate documentation
docs:
	@echo "Generating documentation..."
	@mkdir -p docs/api
	pdoc --html --output-dir docs/api src

# Development setup
dev-setup: install
	pip install -r requirements-dev.txt
	pre-commit install

# Run pre-commit checks
pre-commit:
	pre-commit run --all-files