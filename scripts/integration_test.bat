@echo off
REM Integration test script for Windows - requires Docker Compose to be running

set BASE_URL=http://localhost:8080
set ANOMALY_URL=http://localhost:5001
set RISK_URL=http://localhost:5002

echo Testing Risk Scoring Platform
echo ==============================
echo.

echo 1. Testing Health Endpoints...

echo    - Gateway: %BASE_URL%/health
curl -s "%BASE_URL%/health"
echo.

echo    - Anomaly Service: %ANOMALY_URL%/health
curl -s "%ANOMALY_URL%/health"
echo.

echo    - Risk Service: %RISK_URL%/health
curl -s "%RISK_URL%/health"
echo.

echo.
echo 2. Testing Transaction Processing...

set TRANSACTION_ID=test-%RANDOM%
set PAYLOAD={"transaction_id": "%TRANSACTION_ID%", "user_id": "user-12345", "amount": 1500.00, "currency": "BRL", "country": "BR", "device": "mobile", "timestamp": "2026-01-28T14:30:00Z"}

echo    Sending transaction: %TRANSACTION_ID%
curl -s -X POST "%BASE_URL%/transactions" -H "Content-Type: application/json" -d "%PAYLOAD%"
echo.

echo.
echo 3. Testing High-Risk Transaction...
set HIGH_RISK_PAYLOAD={"transaction_id": "high-risk-%RANDOM%", "user_id": "suspicious-user", "amount": 50000.00, "currency": "BTC", "country": "XX", "device": "unknown", "timestamp": "2026-01-28T03:00:00Z"}
curl -s -X POST "%BASE_URL%/transactions" -H "Content-Type: application/json" -d "%HIGH_RISK_PAYLOAD%"
echo.

echo.
echo ==============================
echo Integration Tests Complete!
echo ==============================
