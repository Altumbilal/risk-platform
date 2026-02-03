//go:build integration
// +build integration

package tests

import (
	"bytes"
	"encoding/json"
	"net/http"
	"testing"
	"time"
)

const baseURL = "http://localhost:8080"

type TransactionRequest struct {
	TransactionID string    `json:"transaction_id"`
	UserID        string    `json:"user_id"`
	Amount        float64   `json:"amount"`
	Currency      string    `json:"currency"`
	Country       string    `json:"country"`
	Device        string    `json:"device"`
	Timestamp     time.Time `json:"timestamp"`
}

type TransactionResponse struct {
	TransactionID   string    `json:"transaction_id"`
	AnomalyScore    float64   `json:"anomaly_score"`
	RiskProbability float64   `json:"risk_probability"`
	Decision        string    `json:"decision"`
	ProcessedAt     time.Time `json:"processed_at"`
}

func TestHealthEndpoint(t *testing.T) {
	resp, err := http.Get(baseURL + "/health")
	if err != nil {
		t.Fatalf("Failed to call health endpoint: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status 200, got %d", resp.StatusCode)
	}
}

func TestTransactionFlow(t *testing.T) {
	txReq := TransactionRequest{
		TransactionID: "integration-test-" + time.Now().Format("20060102150405"),
		UserID:        "test-user-123",
		Amount:        1500.00,
		Currency:      "BRL",
		Country:       "BR",
		Device:        "mobile",
		Timestamp:     time.Now().UTC(),
	}

	body, err := json.Marshal(txReq)
	if err != nil {
		t.Fatalf("Failed to marshal request: %v", err)
	}

	// First request
	resp, err := http.Post(baseURL+"/transactions", "application/json", bytes.NewBuffer(body))
	if err != nil {
		t.Fatalf("Failed to call transactions endpoint: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status 200, got %d", resp.StatusCode)
	}

	var txResp TransactionResponse
	if err := json.NewDecoder(resp.Body).Decode(&txResp); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	// Validate response
	if txResp.TransactionID != txReq.TransactionID {
		t.Errorf("Transaction ID mismatch: expected %s, got %s", txReq.TransactionID, txResp.TransactionID)
	}

	if txResp.AnomalyScore < 0 || txResp.AnomalyScore > 1 {
		t.Errorf("Anomaly score out of range: %v", txResp.AnomalyScore)
	}

	if txResp.RiskProbability < 0 || txResp.RiskProbability > 1 {
		t.Errorf("Risk probability out of range: %v", txResp.RiskProbability)
	}

	validDecisions := map[string]bool{"APPROVE": true, "FLAG": true, "REVIEW": true, "DENY": true}
	if !validDecisions[txResp.Decision] {
		t.Errorf("Invalid decision: %s", txResp.Decision)
	}
}

func TestIdempotency(t *testing.T) {
	txID := "idempotency-test-" + time.Now().Format("20060102150405")
	txReq := TransactionRequest{
		TransactionID: txID,
		UserID:        "test-user-123",
		Amount:        500.00,
		Currency:      "BRL",
		Country:       "BR",
		Device:        "desktop",
		Timestamp:     time.Now().UTC(),
	}

	body, _ := json.Marshal(txReq)

	// First request
	resp1, err := http.Post(baseURL+"/transactions", "application/json", bytes.NewBuffer(body))
	if err != nil {
		t.Fatalf("First request failed: %v", err)
	}
	defer resp1.Body.Close()

	var txResp1 TransactionResponse
	json.NewDecoder(resp1.Body).Decode(&txResp1)

	// Second request with same transaction ID (should return cached result)
	body, _ = json.Marshal(txReq)
	resp2, err := http.Post(baseURL+"/transactions", "application/json", bytes.NewBuffer(body))
	if err != nil {
		t.Fatalf("Second request failed: %v", err)
	}
	defer resp2.Body.Close()

	var txResp2 TransactionResponse
	json.NewDecoder(resp2.Body).Decode(&txResp2)

	// Responses should be identical
	if txResp1.TransactionID != txResp2.TransactionID {
		t.Error("Transaction ID mismatch in idempotent requests")
	}
	if txResp1.AnomalyScore != txResp2.AnomalyScore {
		t.Error("Anomaly score mismatch in idempotent requests")
	}
	if txResp1.RiskProbability != txResp2.RiskProbability {
		t.Error("Risk probability mismatch in idempotent requests")
	}
	if txResp1.Decision != txResp2.Decision {
		t.Error("Decision mismatch in idempotent requests")
	}
}

func TestRiskAnalyzeEndpoint(t *testing.T) {
	txReq := TransactionRequest{
		TransactionID: "analyze-test-" + time.Now().Format("20060102150405"),
		UserID:        "test-user-456",
		Amount:        10000.00, // High amount
		Currency:      "USD",
		Country:       "XX",
		Device:        "unknown",
		Timestamp:     time.Now().UTC(),
	}

	body, _ := json.Marshal(txReq)

	resp, err := http.Post(baseURL+"/risk/analyze", "application/json", bytes.NewBuffer(body))
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status 200, got %d", resp.StatusCode)
	}

	var txResp TransactionResponse
	if err := json.NewDecoder(resp.Body).Decode(&txResp); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	// Validate scores are in range
	if txResp.AnomalyScore < 0 || txResp.AnomalyScore > 1 {
		t.Errorf("Anomaly score out of range: %v", txResp.AnomalyScore)
	}

	if txResp.RiskProbability < 0 || txResp.RiskProbability > 1 {
		t.Errorf("Risk probability out of range: %v", txResp.RiskProbability)
	}
}
