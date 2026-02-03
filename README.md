# Real-Time ML Risk Scoring Platform

<div align="center">

![Go](https://img.shields.io/badge/Go-1.22+-00ADD8?style=for-the-badge&logo=go&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-7-DC382D?style=for-the-badge&logo=redis&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**A high-performance, distributed system for real-time transaction risk assessment using ensemble machine learning models.**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Tech Stack](#-tech-stack)

</div>

---

## Overview

This platform analyzes financial transactions in **real-time** using a combination of **anomaly detection** (PyTorch Autoencoder) and **risk classification** (TensorFlow Neural Network) to make automated risk decisions:

|  Decision   | Description                                   |
| :---------: | --------------------------------------------- |
| **APPROVE** | Transaction appears safe - proceed normally   |
|  **FLAG**   | Elevated risk detected - requires monitoring  |
| **REVIEW**  | High risk indicators - manual review required |

The system processes transactions in **milliseconds**, combining scores from multiple ML models with a configurable decision engine, while maintaining full auditability through PostgreSQL persistence.

---

## Features

### Core Capabilities

- **Real-Time Processing** — Sub-second latency with parallel ML inference
- **Ensemble ML Models** — Combines anomaly detection + risk classification for robust scoring
- **Smart Decision Engine** — Threshold-based decisions with weighted scoring (40% anomaly / 60% risk)
- **Idempotency** — Redis-backed deduplication prevents double-processing
- **Full Audit Trail** — Every transaction persisted to PostgreSQL with complete metadata
- **Container-Ready** — Single command deployment with Docker Compose

### Machine Learning

- **Autoencoder Neural Network** (PyTorch) — Learns normal transaction patterns, flags deviations
- **Binary Classifier** (TensorFlow) — Deep neural network trained on labeled fraud data
- **Rich Feature Engineering** — 20+ engineered features including temporal patterns, risk scores, and domain-specific indicators

### Production-Ready

- **Graceful Degradation** — Services fail safely with neutral fallback scores
- **Connection Pooling** — Optimized database and cache connections
- **Health Checks** — Built-in monitoring endpoints for all services
- **Multi-stage Docker Builds** — Minimal, secure container images

---

## Architecture

```
                              CLIENT REQUEST
                          POST /transactions
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GO API GATEWAY (Gin)                               │
│                              Port: 8080                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  • Request Validation    • Feature Extraction    • Parallel Calls   │   │
│  │  • Redis Caching         • Decision Engine       • Response Format  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────┬─────────────────────────┬──────────────────────────┘
                         │                         │
            ┌────────────┘                         └────────────┐
            │  PARALLEL                               PARALLEL  │
            ▼                                                   ▼
┌───────────────────────────────┐         ┌───────────────────────────────────┐
│   ANOMALY DETECTION SERVICE   │         │     RISK CLASSIFICATION SERVICE   │
│        (PyTorch + Flask)      │         │       (TensorFlow + Flask)        │
│          Port: 5001           │         │            Port: 5002             │
│  ┌─────────────────────────┐  │         │  ┌─────────────────────────────┐  │
│  │    AUTOENCODER NN       │  │         │  │     BINARY CLASSIFIER       │  │
│  │  ┌─────────────────┐    │  │         │  │  ┌─────────────────────┐    │  │
│  │  │ Encoder → Latent│    │  │         │  │  │ BatchNorm → Dense   │    │  │
│  │  │ Latent → Decoder│    │  │         │  │  │ Dense → Dense       │    │  │
│  │  │ Reconstruction  │    │  │         │  │  │ Dense → Sigmoid     │    │  │
│  │  │ Error = Score   │    │  │         │  │  │ Output = Prob[0,1]  │    │  │
│  │  └─────────────────┘    │  │         │  │  └─────────────────────┘    │  │
│  └─────────────────────────┘  │         │  └─────────────────────────────┘  │
└───────────────────────────────┘         └───────────────────────────────────┘
            │                                                   │
            │         anomaly_score: 0.0 - 1.0                  │
            │         risk_probability: 0.0 - 1.0               │
            └────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
            ┌────────────────────────────────────────┐
            │           DECISION ENGINE              │
            │  ┌──────────────────────────────────┐  │
            │  │ if anomaly > 0.8 OR risk > 0.75  │  │
            │  │     → REVIEW                     │  │
            │  │ elif anomaly > 0.5 OR risk > 0.5 │  │
            │  │     → FLAG                       │  │
            │  │ else                             │  │
            │  │     → APPROVE                    │  │
            │  └──────────────────────────────────┘  │
            └────────────────────────────────────────┘
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
┌───────────────────────┐               ┌───────────────────────┐
│    POSTGRESQL 16      │               │        REDIS 7        │
│    (Persistence)      │               │        (Cache)        │
│  • transactions table │               │  • Idempotency keys   │
│  • JSONB payloads     │               │  • 24h TTL            │
│  • Full audit trail   │               │  • Connection pool    │
└───────────────────────┘               └───────────────────────┘
```

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Git

### One-Command Deployment

```bash
# Clone the repository
git clone https://github.com/yourusername/risk-platform.git
cd risk-platform

# Start all services
docker-compose up -d

# Verify all services are healthy
docker-compose ps
```

The platform will be available at `http://localhost:8080`

### Test a Transaction

```bash
curl -X POST http://localhost:8080/transactions \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx-demo-001",
    "user_id": "user-123",
    "amount": 150.00,
    "currency": "USD",
    "country": "US",
    "device": "mobile",
    "timestamp": "2026-01-28T14:30:00Z"
  }'
```

**Response:**

```json
{
  "transaction_id": "tx-demo-001",
  "anomaly_score": 0.23,
  "risk_probability": 0.31,
  "combined_score": 0.28,
  "decision": "APPROVE",
  "processed_at": "2026-01-28T14:30:01Z"
}
```

---

## API Reference

### Endpoints

| Method | Endpoint         | Description                          |
| ------ | ---------------- | ------------------------------------ |
| `GET`  | `/health`        | Health check for all services        |
| `POST` | `/transactions`  | Process transaction with persistence |
| `POST` | `/risk/analyze`  | Analyze without persisting           |
| `POST` | `/anomaly/score` | Anomaly detection only               |
| `POST` | `/risk/classify` | Risk classification only             |
| `POST` | `/risk/batch`    | Batch classification                 |
| `GET`  | `/model/info`    | ML model metadata                    |

### Request Schema

```json
{
  "transaction_id": "string",
  "user_id": "string",
  "amount": 1500.0,
  "currency": "USD",
  "country": "US",
  "device": "mobile",
  "timestamp": "2026-01-28T14:30:00Z"
}
```

### Response Schema

```json
{
  "transaction_id": "string",
  "anomaly_score": 0.23,
  "risk_probability": 0.31,
  "combined_score": 0.28,
  "decision": "APPROVE",
  "processed_at": "2026-01-28T14:30:01Z"
}
```

---

## ML Models Deep Dive

### Anomaly Detection (PyTorch Autoencoder)

The autoencoder learns a compressed representation of **normal** transaction patterns. During inference, it attempts to reconstruct the input — high reconstruction error indicates anomalous behavior.

```
Input (10D) → Encoder [64→32→16] → Latent (16D) → Decoder [16→32→64] → Output (10D)
                                                                           │
                                              Reconstruction Error ────────┘
                                              (MSE between input & output)
```

**Features Used:**

- Normalized amount (z-score)
- Hour & day-of-week (cyclical encoding)
- Device type encoding
- Country risk score
- One-hot currency (4 dimensions)

### Risk Classification (TensorFlow Neural Network)

A binary classifier trained on labeled transaction data, predicting the probability of a transaction being fraudulent.

```
Input (12D) → BatchNorm → Dense(64, ReLU) → Dropout(0.3)
                                              ↓
           Output ← Sigmoid ← Dense(16) ← Dense(32, ReLU)
```

**Features Used:**

- Amount: z-score, log-normalized, size indicators
- Temporal: hour normalization, night flag, business hours, weekend
- Geographic: country risk score, high/medium risk flags
- Device: risk score, mobile flag, unknown flag
- Currency: risk score, crypto flag, major currency flag

---

## Project Structure

```
risk-platform/
├── gateway-go/                     # Go API Gateway
│   ├── cmd/main.go                 # Application entrypoint
│   ├── internal/
│   │   ├── api/handler.go          # HTTP handlers & routing
│   │   ├── decision/engine.go      # Decision logic
│   │   ├── models/transaction.go   # Data structures
│   │   ├── services/               # ML service clients
│   │   └── storage/                # PostgreSQL & Redis
│   └── Dockerfile
│
├── ml-anomaly-pytorch/             # Anomaly Detection Service
│   ├── model/
│   │   ├── autoencoder.py          # Neural network architecture
│   │   └── preprocessing.py        # Feature engineering
│   ├── inference/server.py         # Flask inference server
│   ├── train.py                    # Training script
│   └── Dockerfile
│
├── ml-risk-tensorflow/             # Risk Classification Service
│   ├── model/
│   │   ├── classifier.py           # Classifier architecture
│   │   └── preprocessing.py        # Feature engineering
│   ├── inference/server.py         # Flask inference server
│   ├── train.py                    # Training script
│   └── Dockerfile
│
├── docker-compose.yml              # Full stack orchestration
├── init.sql                        # PostgreSQL schema
├── postman_collection.json         # API testing collection
└── Makefile                        # Development commands
```

---

## Tech Stack

| Layer                   | Technology               | Purpose                         |
| ----------------------- | ------------------------ | ------------------------------- |
| **API Gateway**         | Go 1.22+ / Gin           | High-performance HTTP routing   |
| **Anomaly Detection**   | Python / PyTorch 2.0     | Autoencoder neural network      |
| **Risk Classification** | Python / TensorFlow 2.14 | Binary classifier               |
| **Inference Servers**   | Flask / Gunicorn         | Production-ready WSGI           |
| **Primary Database**    | PostgreSQL 16            | Transaction persistence & audit |
| **Cache**               | Redis 7                  | Idempotency & performance       |
| **Containerization**    | Docker / Compose         | Deployment & orchestration      |

---

## Database Schema

### PostgreSQL - transactions table

| Column         | Type      | Description                               |
| -------------- | --------- | ----------------------------------------- |
| id             | UUID      | Primary key                               |
| transaction_id | VARCHAR   | External transaction ID (idempotency key) |
| payload        | JSONB     | Original transaction data                 |
| anomaly_score  | FLOAT     | Anomaly detection score [0-1]             |
| risk_score     | FLOAT     | Risk classification score [0-1]           |
| decision       | VARCHAR   | Final decision (APPROVE/FLAG/REVIEW)      |
| created_at     | TIMESTAMP | Processing timestamp                      |

**Indexes:** transaction_id, created_at, decision, risk_score

**Views:** `high_risk_transactions` - Pre-filtered view for flagged/review transactions

---

## Testing

```bash
# Run Go tests
cd gateway-go && go test ./...

# Run Python tests
cd ml-anomaly-pytorch && pytest
cd ml-risk-tensorflow && pytest

# Integration test (requires running services)
./scripts/integration_test.sh
```

---

## Performance Characteristics

| Metric             | Value                    |
| ------------------ | ------------------------ |
| **Latency (p50)**  | ~50ms                    |
| **Latency (p99)**  | ~150ms                   |
| **Throughput**     | 1000+ req/s per instance |
| **ML Inference**   | ~20ms per model          |
| **Cache Hit Rate** | 24h TTL for idempotency  |

---

## Future Enhancements

- [ ] **Model Monitoring** — Drift detection and automated retraining pipelines
- [ ] **A/B Testing** — Experiment with different model versions in production
- [ ] **gRPC Support** — Higher performance inter-service communication
- [ ] **Kubernetes** — Production-grade orchestration with horizontal pod autoscaling
- [ ] **Feature Store** — Centralized feature management and versioning
- [ ] **Explainability** — SHAP/LIME integration for decision transparency

---

## License

MIT License

---

<div align="center">

**Built with Go, Python, PyTorch, TensorFlow & Docker**

</div>
