# Makefile for Risk Platform

.PHONY: all build up down test clean logs

# Default target
all: build up

# Build all services
build:
	docker-compose build

# Start all services
up:
	docker-compose up -d

# Start with logs
up-logs:
	docker-compose up

# Stop all services
down:
	docker-compose down

# Stop and remove volumes
down-clean:
	docker-compose down -v

# View logs
logs:
	docker-compose logs -f

# View logs for specific service
logs-gateway:
	docker-compose logs -f gateway

logs-anomaly:
	docker-compose logs -f ml-anomaly

logs-risk:
	docker-compose logs -f ml-risk

# Run tests locally
test:
	@echo "Running Go tests..."
	cd gateway-go && go test ./... -v
	@echo ""
	@echo "Running PyTorch service tests..."
	cd ml-anomaly-pytorch && python -m pytest tests/ -v
	@echo ""
	@echo "Running TensorFlow service tests..."
	cd ml-risk-tensorflow && python -m pytest tests/ -v

# Run Go tests only
test-go:
	cd gateway-go && go test ./... -v

# Run Python tests only
test-python:
	cd ml-anomaly-pytorch && python -m pytest tests/ -v
	cd ml-risk-tensorflow && python -m pytest tests/ -v

# Run integration tests (requires services to be running)
test-integration:
	./scripts/integration_test.sh

# Clean up
clean:
	docker-compose down -v
	docker system prune -f

# Train models
train-anomaly:
	cd ml-anomaly-pytorch && python train.py --epochs 50

train-risk:
	cd ml-risk-tensorflow && python train.py --epochs 50

train-all: train-anomaly train-risk

# Development - start dependencies only
dev-deps:
	docker-compose up -d postgres redis

# Development - format code
fmt:
	cd gateway-go && go fmt ./...

# Development - lint Go code
lint-go:
	cd gateway-go && golangci-lint run

# Health check
health:
	@curl -s http://localhost:8080/health | jq .
	@curl -s http://localhost:5001/health | jq .
	@curl -s http://localhost:5002/health | jq .
