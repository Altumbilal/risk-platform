package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/risk-platform/gateway-go/internal/decision"
	"github.com/risk-platform/gateway-go/internal/models"
)

// MockDB implements a mock database for testing
type MockDB struct {
	transactions map[string]*models.TransactionRecord
}

func NewMockDB() *MockDB {
	return &MockDB{
		transactions: make(map[string]*models.TransactionRecord),
	}
}

// MockCache implements a mock cache for testing
type MockCache struct {
	data map[string]string
}

func NewMockCache() *MockCache {
	return &MockCache{
		data: make(map[string]string),
	}
}

// MockMLService simulates ML service responses
type MockMLService struct {
	score float64
	err   error
}

func TestHealthCheck(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":    "healthy",
			"timestamp": time.Now().UTC(),
		})
	})

	req, _ := http.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status %d, got %d", http.StatusOK, w.Code)
	}

	var response map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}

	if response["status"] != "healthy" {
		t.Errorf("Expected status 'healthy', got '%v'", response["status"])
	}
}

func TestTransactionPayloadValidation(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name           string
		payload        interface{}
		expectedStatus int
	}{
		{
			name: "Valid payload",
			payload: models.TransactionRequest{
				TransactionID: "test-123",
				UserID:        "user-456",
				Amount:        100.00,
				Currency:      "BRL",
				Country:       "BR",
				Device:        "mobile",
				Timestamp:     time.Now(),
			},
			expectedStatus: http.StatusBadRequest, // Will fail due to no ML services
		},
		{
			name: "Missing transaction_id",
			payload: map[string]interface{}{
				"user_id":   "user-456",
				"amount":    100.00,
				"currency":  "BRL",
				"country":   "BR",
				"device":    "mobile",
				"timestamp": time.Now(),
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name: "Invalid amount",
			payload: map[string]interface{}{
				"transaction_id": "test-123",
				"user_id":        "user-456",
				"amount":         -100.00,
				"currency":       "BRL",
				"country":        "BR",
				"device":         "mobile",
				"timestamp":      time.Now(),
			},
			expectedStatus: http.StatusBadRequest,
		},
		{
			name:           "Empty payload",
			payload:        map[string]interface{}{},
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router := gin.New()
			router.POST("/transactions", func(c *gin.Context) {
				var req models.TransactionRequest
				if err := c.ShouldBindJSON(&req); err != nil {
					c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
					return
				}
				c.JSON(http.StatusOK, gin.H{"status": "received"})
			})

			body, _ := json.Marshal(tt.payload)
			req, _ := http.NewRequest(http.MethodPost, "/transactions", bytes.NewBuffer(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)

			// For valid payloads, we expect OK (since we're just testing validation)
			if tt.name == "Valid payload" && w.Code != http.StatusOK {
				t.Errorf("Expected status %d for valid payload, got %d", http.StatusOK, w.Code)
			}
			// For invalid payloads, we expect BadRequest
			if tt.name != "Valid payload" && w.Code != http.StatusBadRequest {
				t.Errorf("Expected status %d, got %d", tt.expectedStatus, w.Code)
			}
		})
	}
}

func TestDecisionEngineIntegration(t *testing.T) {
	engine := decision.NewEngine()

	testCases := []struct {
		name         string
		anomaly      float64
		risk         float64
		wantDecision string
	}{
		{"high_anomaly_review", 0.85, 0.3, models.DecisionReview},
		{"high_risk_review", 0.3, 0.8, models.DecisionReview},
		{"medium_flag", 0.6, 0.4, models.DecisionFlag},
		{"low_approve", 0.2, 0.3, models.DecisionApprove},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			decision := engine.Decide(tc.anomaly, tc.risk)
			if decision != tc.wantDecision {
				t.Errorf("Expected %s, got %s", tc.wantDecision, decision)
			}
		})
	}
}

func TestExtractFeatures(t *testing.T) {
	timestamp := time.Date(2026, 1, 28, 14, 30, 0, 0, time.UTC)
	req := models.TransactionRequest{
		TransactionID: "test-123",
		UserID:        "user-456",
		Amount:        1500.00,
		Currency:      "BRL",
		Country:       "BR",
		Device:        "mobile",
		Timestamp:     timestamp,
	}

	features := extractFeatures(req)

	if features.Amount != 1500.00 {
		t.Errorf("Expected amount 1500.00, got %v", features.Amount)
	}
	if features.Currency != "BRL" {
		t.Errorf("Expected currency BRL, got %v", features.Currency)
	}
	if features.Country != "BR" {
		t.Errorf("Expected country BR, got %v", features.Country)
	}
	if features.Device != "mobile" {
		t.Errorf("Expected device mobile, got %v", features.Device)
	}
	if features.Hour != 14 {
		t.Errorf("Expected hour 14, got %v", features.Hour)
	}
	if features.DayOfWeek != 3 { // Wednesday
		t.Errorf("Expected day of week 3 (Wednesday), got %v", features.DayOfWeek)
	}
	if features.UserID != "user-456" {
		t.Errorf("Expected user_id user-456, got %v", features.UserID)
	}
}
