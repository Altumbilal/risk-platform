#!/bin/bash
# Integration test script - requires Docker Compose to be running

set -e

BASE_URL=${BASE_URL:-"http://localhost:8080"}
ANOMALY_URL=${ANOMALY_URL:-"http://localhost:5001"}
RISK_URL=${RISK_URL:-"http://localhost:5002"}

echo "Testing Risk Scoring Platform"
echo "=============================="
echo ""

# Test health endpoints
echo "1. Testing Health Endpoints..."

echo "   - Gateway: $BASE_URL/health"
curl -s "$BASE_URL/health" | jq .
echo ""

echo "   - Anomaly Service: $ANOMALY_URL/health"
curl -s "$ANOMALY_URL/health" | jq .
echo ""

echo "   - Risk Service: $RISK_URL/health"
curl -s "$RISK_URL/health" | jq .
echo ""

# Test transaction processing
echo "2. Testing Transaction Processing..."

TRANSACTION_ID="test-$(date +%s)"
PAYLOAD='{
  "transaction_id": "'$TRANSACTION_ID'",
  "user_id": "user-12345",
  "amount": 1500.00,
  "currency": "BRL",
  "country": "BR",
  "device": "mobile",
  "timestamp": "2026-01-28T14:30:00Z"
}'

echo "   Sending transaction: $TRANSACTION_ID"
RESPONSE=$(curl -s -X POST "$BASE_URL/transactions" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

echo "   Response:"
echo "$RESPONSE" | jq .
echo ""

# Test idempotency
echo "3. Testing Idempotency (same transaction ID)..."
RESPONSE2=$(curl -s -X POST "$BASE_URL/transactions" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

echo "   Response (should be identical):"
echo "$RESPONSE2" | jq .
echo ""

# Test high-risk transaction
echo "4. Testing High-Risk Transaction..."
HIGH_RISK_PAYLOAD='{
  "transaction_id": "high-risk-'$(date +%s)'",
  "user_id": "suspicious-user",
  "amount": 50000.00,
  "currency": "BTC",
  "country": "XX",
  "device": "unknown",
  "timestamp": "2026-01-28T03:00:00Z"
}'

RESPONSE3=$(curl -s -X POST "$BASE_URL/transactions" \
  -H "Content-Type: application/json" \
  -d "$HIGH_RISK_PAYLOAD")

echo "   Response (should be REVIEW or higher risk):"
echo "$RESPONSE3" | jq .
echo ""

# Test risk analysis endpoint
echo "5. Testing Risk Analysis Endpoint..."
RESPONSE4=$(curl -s -X POST "$BASE_URL/risk/analyze" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

echo "   Response:"
echo "$RESPONSE4" | jq .
echo ""

# Test ML service endpoints directly
echo "6. Testing ML Services Directly..."

echo "   - Anomaly Detection:"
curl -s -X POST "$ANOMALY_URL/anomaly/score" \
  -H "Content-Type: application/json" \
  -d '{"amount": 1500, "currency": "BRL", "country": "BR", "device": "mobile", "hour": 14, "day_of_week": 3}' | jq .
echo ""

echo "   - Risk Classification:"
curl -s -X POST "$RISK_URL/risk/classify" \
  -H "Content-Type: application/json" \
  -d '{"amount": 1500, "currency": "BRL", "country": "BR", "device": "mobile", "hour": 14, "day_of_week": 3}' | jq .
echo ""

echo "=============================="
echo "Integration Tests Complete!"
echo "=============================="
