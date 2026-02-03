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

// AnomalyService handles communication with PyTorch anomaly detection service
type AnomalyService struct {
	baseURL    string
	httpClient *http.Client
}

// NewAnomalyService creates a new anomaly service client
func NewAnomalyService(baseURL string) *AnomalyService {
	return &AnomalyService{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
		},
	}
}

// Score calls the anomaly detection endpoint and returns the anomaly score
func (s *AnomalyService) Score(ctx context.Context, features models.MLFeatures) (float64, error) {
	url := fmt.Sprintf("%s/anomaly/score", s.baseURL)

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
		return 0, fmt.Errorf("failed to call anomaly service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("anomaly service returned status %d", resp.StatusCode)
	}

	var result models.AnomalyResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return 0, fmt.Errorf("failed to decode response: %w", err)
	}

	return result.AnomalyScore, nil
}

// Health checks if the anomaly service is healthy
func (s *AnomalyService) Health(ctx context.Context) error {
	url := fmt.Sprintf("%s/health", s.baseURL)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := s.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to call anomaly service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("anomaly service unhealthy: status %d", resp.StatusCode)
	}

	return nil
}
