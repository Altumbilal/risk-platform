package main

import (
	"log"
	"os"

	"github.com/gin-gonic/gin"
	"github.com/risk-platform/gateway-go/internal/api"
	"github.com/risk-platform/gateway-go/internal/decision"
	"github.com/risk-platform/gateway-go/internal/services"
	"github.com/risk-platform/gateway-go/internal/storage"
)

func main() {
	// Configuration from environment
	config := &Config{
		Port:              getEnv("PORT", "8080"),
		PostgresURL:       getEnv("POSTGRES_URL", "postgres://postgres:postgres@localhost:5432/riskdb?sslmode=disable"),
		RedisURL:          getEnv("REDIS_URL", "localhost:6379"),
		AnomalyServiceURL: getEnv("ANOMALY_SERVICE_URL", "http://localhost:5001"),
		RiskServiceURL:    getEnv("RISK_SERVICE_URL", "http://localhost:5002"),
	}

	// Initialize storage
	db, err := storage.NewPostgresDB(config.PostgresURL)
	if err != nil {
		log.Fatalf("Failed to connect to PostgreSQL: %v", err)
	}
	defer db.Close()

	cache, err := storage.NewRedisCache(config.RedisURL)
	if err != nil {
		log.Fatalf("Failed to connect to Redis: %v", err)
	}
	defer cache.Close()

	// Initialize ML services
	anomalyService := services.NewAnomalyService(config.AnomalyServiceURL)
	riskService := services.NewRiskService(config.RiskServiceURL)

	// Initialize decision engine
	decisionEngine := decision.NewEngine()

	// Initialize API handler
	handler := api.NewHandler(db, cache, anomalyService, riskService, decisionEngine)

	// Setup router
	router := gin.Default()
	handler.RegisterRoutes(router)

	// Start server
	log.Printf("Starting API Gateway on port %s", config.Port)
	if err := router.Run(":" + config.Port); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

type Config struct {
	Port              string
	PostgresURL       string
	RedisURL          string
	AnomalyServiceURL string
	RiskServiceURL    string
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
