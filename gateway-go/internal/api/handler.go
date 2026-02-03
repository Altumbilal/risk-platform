package api

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/risk-platform/gateway-go/internal/decision"
	"github.com/risk-platform/gateway-go/internal/models"
	"github.com/risk-platform/gateway-go/internal/services"
	"github.com/risk-platform/gateway-go/internal/storage"
)

// Handler handles HTTP requests
type Handler struct {
	db             *storage.PostgresDB
	cache          *storage.RedisCache
	anomalyService *services.AnomalyService
	riskService    *services.RiskService
	decisionEngine *decision.Engine
}

// NewHandler creates a new API handler
func NewHandler(
	db *storage.PostgresDB,
	cache *storage.RedisCache,
	anomalyService *services.AnomalyService,
	riskService *services.RiskService,
	decisionEngine *decision.Engine,
) *Handler {
	return &Handler{
		db:             db,
		cache:          cache,
		anomalyService: anomalyService,
		riskService:    riskService,
		decisionEngine: decisionEngine,
	}
}

// RegisterRoutes registers all API routes
func (h *Handler) RegisterRoutes(router *gin.Engine) {
	router.GET("/health", h.HealthCheck)
	router.POST("/transactions", h.ProcessTransaction)
	router.POST("/risk/analyze", h.AnalyzeRisk)
}

// HealthCheck returns the service health status
func (h *Handler) HealthCheck(c *gin.Context) {
	status := map[string]interface{}{
		"status":    "healthy",
		"timestamp": time.Now().UTC(),
		"services": map[string]string{
			"database": "connected",
			"cache":    "connected",
		},
	}
	c.JSON(http.StatusOK, status)
}

// ProcessTransaction handles transaction processing
func (h *Handler) ProcessTransaction(c *gin.Context) {
	var req models.TransactionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Check idempotency
	ctx := context.Background()
	cached, err := h.cache.Get(ctx, req.TransactionID)
	if err == nil && cached != "" {
		var cachedResponse models.TransactionResponse
		if json.Unmarshal([]byte(cached), &cachedResponse) == nil {
			c.JSON(http.StatusOK, cachedResponse)
			return
		}
	}

	// Extract features for ML services
	features := extractFeatures(req)

	// Call ML services in parallel
	var anomalyScore, riskProbability float64
	var anomalyErr, riskErr error
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		anomalyScore, anomalyErr = h.anomalyService.Score(ctx, features)
	}()

	go func() {
		defer wg.Done()
		riskProbability, riskErr = h.riskService.Classify(ctx, features)
	}()

	wg.Wait()

	// Handle ML service errors
	if anomalyErr != nil {
		log.Printf("Anomaly service error: %v", anomalyErr)
		anomalyScore = 0.5 // Default fallback
	}
	if riskErr != nil {
		log.Printf("Risk service error: %v", riskErr)
		riskProbability = 0.5 // Default fallback
	}

	// Make decision
	decisionResult := h.decisionEngine.Decide(anomalyScore, riskProbability)

	// Build response
	response := models.TransactionResponse{
		TransactionID:   req.TransactionID,
		AnomalyScore:    anomalyScore,
		RiskProbability: riskProbability,
		Decision:        decisionResult,
		ProcessedAt:     time.Now().UTC(),
	}

	// Persist to database
	payloadJSON, _ := json.Marshal(req)
	record := &models.TransactionRecord{
		ID:            uuid.New().String(),
		TransactionID: req.TransactionID,
		Payload:       string(payloadJSON),
		AnomalyScore:  anomalyScore,
		RiskScore:     riskProbability,
		Decision:      decisionResult,
		CreatedAt:     time.Now().UTC(),
	}

	if err := h.db.SaveTransaction(ctx, record); err != nil {
		log.Printf("Failed to save transaction: %v", err)
	}

	// Cache the response for idempotency
	responseJSON, _ := json.Marshal(response)
	if err := h.cache.Set(ctx, req.TransactionID, string(responseJSON), 24*time.Hour); err != nil {
		log.Printf("Failed to cache response: %v", err)
	}

	c.JSON(http.StatusOK, response)
}

// AnalyzeRisk performs risk analysis without persisting
func (h *Handler) AnalyzeRisk(c *gin.Context) {
	var req models.TransactionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ctx := context.Background()
	features := extractFeatures(req)

	// Call ML services in parallel
	var anomalyScore, riskProbability float64
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		anomalyScore, _ = h.anomalyService.Score(ctx, features)
	}()

	go func() {
		defer wg.Done()
		riskProbability, _ = h.riskService.Classify(ctx, features)
	}()

	wg.Wait()

	decisionResult := h.decisionEngine.Decide(anomalyScore, riskProbability)

	response := models.TransactionResponse{
		TransactionID:   req.TransactionID,
		AnomalyScore:    anomalyScore,
		RiskProbability: riskProbability,
		Decision:        decisionResult,
		ProcessedAt:     time.Now().UTC(),
	}

	c.JSON(http.StatusOK, response)
}

// extractFeatures converts transaction request to ML features
func extractFeatures(req models.TransactionRequest) models.MLFeatures {
	return models.MLFeatures{
		Amount:    req.Amount,
		Currency:  req.Currency,
		Country:   req.Country,
		Device:    req.Device,
		Hour:      req.Timestamp.Hour(),
		DayOfWeek: int(req.Timestamp.Weekday()),
		UserID:    req.UserID,
	}
}
