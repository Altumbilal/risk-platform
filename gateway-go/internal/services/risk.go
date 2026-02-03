package services

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/risk-platform/gateway-go/internal/models"
)

// RiskService handles communication with TensorFlow risk classification service
type RiskService struct {
	baseURL    string
	httpClient *http.Client
}

// NewRiskService creates a new risk service client
func NewRiskService(baseURL string) *RiskService {
	return &RiskService{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
		},
	}
}

// Classify calls the risk classification endpoint and returns the risk probability
func (s *RiskService) Classify(ctx context.Context, features models.MLFeatures) (float64, error) {
	url := fmt.Sprintf("%s/risk/classify", s.baseURL)

	payload, err := json.Marshal(features)
	if err != nil {
		return 0, fmt.Errorf("failed to marshal features: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(payload))
	if err != nil {
		return 0, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return 0, fmt.Errorf("failed to call risk service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("risk service returned status %d", resp.StatusCode)
	}

	var result models.RiskResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return 0, fmt.Errorf("failed to decode response: %w", err)
	}

	return result.RiskProbability, nil
}

// Health checks if the risk service is healthy
func (s *RiskService) Health(ctx context.Context) error {
	url := fmt.Sprintf("%s/health", s.baseURL)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to call risk service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("risk service unhealthy: status %d", resp.StatusCode)
	}

	return nil
}
