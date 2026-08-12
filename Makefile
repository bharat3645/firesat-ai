.PHONY: install dev-install data train serve test lint docker-build docker-up clean

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

data:
	python scripts/generate_demo_data.py

train:
	python scripts/train_demo.py

serve:
	uvicorn firesat.api.main:app --reload

test:
	pytest -q

lint:
	ruff check src tests scripts

docker-build:
	docker build -t firesat-ai .

docker-up:
	docker compose up --build

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} \;
	rm -rf .pytest_cache .ruff_cache
