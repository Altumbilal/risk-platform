package models

import (
	"time"
)

// TransactionRequest represents an incoming transaction
type TransactionRequest struct {
	TransactionID string    `json:"transaction_id" binding:"required"`
	UserID        string    `json:"user_id" binding:"required"`
	Amount        float64   `json:"amount" binding:"required,gt=0"`
	Currency      string    `json:"currency" binding:"required"`
	Country       string    `json:"country" binding:"required"`
	Device        string    `json:"device" binding:"required"`
	Timestamp     time.Time `json:"timestamp" binding:"required"`
}

// TransactionResponse represents the API response
type TransactionResponse struct {
	TransactionID   string    `json:"transaction_id"`
	AnomalyScore    float64   `json:"anomaly_score"`
	RiskProbability float64   `json:"risk_probability"`
	Decision        string    `json:"decision"`
	ProcessedAt     time.Time `json:"processed_at"`
}

// TransactionRecord represents a database record
type TransactionRecord struct {
	ID            string    `json:"id"`
	TransactionID string    `json:"transaction_id"`
	Payload       string    `json:"payload"`
	AnomalyScore  float64   `json:"anomaly_score"`
	RiskScore     float64   `json:"risk_score"`
	Decision      string    `json:"decision"`
	CreatedAt     time.Time `json:"created_at"`
}

// MLFeatures represents features sent to ML services
type MLFeatures struct {
	Amount    float64 `json:"amount"`
	Currency  string  `json:"currency"`
	Country   string  `json:"country"`
	Device    string  `json:"device"`
	Hour      int     `json:"hour"`
	DayOfWeek int     `json:"day_of_week"`
	UserID    string  `json:"user_id"`
}

// AnomalyResponse from PyTorch service
type AnomalyResponse struct {
	AnomalyScore float64 `json:"anomaly_score"`
}

// RiskResponse from TensorFlow service
type RiskResponse struct {
	RiskProbability float64 `json:"risk_probability"`
}

// Decision types
const (
	DecisionApprove = "APPROVE"
	DecisionFlag    = "FLAG"
	DecisionReview  = "REVIEW"
	DecisionDeny    = "DENY"
)
