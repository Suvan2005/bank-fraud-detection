.PHONY: install pipeline api dashboard test docker-build clean help

help:
	@echo ""
	@echo "  Banking Fraud Detection — Project Commands"
	@echo "  ────────────────────────────────────────────"
	@echo "  make install       Install Python dependencies"
	@echo "  make pipeline      Run full ML pipeline (data → train → artifacts)"
	@echo "  make api           Start FastAPI server on :8000"
	@echo "  make dashboard     Start Streamlit dashboard on :8501"
	@echo "  make test          Run pytest test suite"
	@echo "  make docker-build  Build Docker image"
	@echo "  make docker-up     Spin up all services with Docker Compose"
	@echo "  make clean         Remove generated data, model artifacts, and logs"
	@echo ""

install:
	pip install -r requirements.txt
	pip install -e .

pipeline:
	python scripts/run_pipeline.py

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	streamlit run dashboard/app.py --server.port 8501

test:
	pytest tests/ -v

docker-build:
	docker build -t bank-fraud-detection:latest .

docker-up:
	docker-compose up --build

clean:
	rm -rf data/raw/*.csv data/processed/*.csv models/*.joblib models/*.json mlruns/ *.log
