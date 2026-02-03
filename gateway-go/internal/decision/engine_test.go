package decision

import (
	"testing"

	"github.com/risk-platform/gateway-go/internal/models"
)

func TestEngine_Decide(t *testing.T) {
	engine := NewEngine()

	tests := []struct {
		name            string
		anomalyScore    float64
		riskProbability float64
		expected        string
	}{
		{
			name:            "High anomaly triggers REVIEW",
			anomalyScore:    0.85,
			riskProbability: 0.3,
			expected:        models.DecisionReview,
		},
		{
			name:            "High risk triggers REVIEW",
			anomalyScore:    0.3,
			riskProbability: 0.8,
			expected:        models.DecisionReview,
		},
		{
			name:            "Both high triggers REVIEW",
			anomalyScore:    0.9,
			riskProbability: 0.9,
			expected:        models.DecisionReview,
		},
		{
			name:            "Medium anomaly triggers FLAG",
			anomalyScore:    0.6,
			riskProbability: 0.3,
			expected:        models.DecisionFlag,
		},
		{
			name:            "Medium risk triggers FLAG",
			anomalyScore:    0.3,
			riskProbability: 0.6,
			expected:        models.DecisionFlag,
		},
		{
			name:            "Both medium triggers FLAG",
			anomalyScore:    0.6,
			riskProbability: 0.6,
			expected:        models.DecisionFlag,
		},
		{
			name:            "Low scores trigger APPROVE",
			anomalyScore:    0.2,
			riskProbability: 0.3,
			expected:        models.DecisionApprove,
		},
		{
			name:            "Zero scores trigger APPROVE",
			anomalyScore:    0.0,
			riskProbability: 0.0,
			expected:        models.DecisionApprove,
		},
		{
			name:            "Boundary - exactly at high anomaly threshold",
			anomalyScore:    0.8,
			riskProbability: 0.3,
			expected:        models.DecisionFlag,
		},
		{
			name:            "Boundary - just above high anomaly threshold",
			anomalyScore:    0.81,
			riskProbability: 0.3,
			expected:        models.DecisionReview,
		},
		{
			name:            "Boundary - exactly at high risk threshold",
			anomalyScore:    0.3,
			riskProbability: 0.75,
			expected:        models.DecisionFlag,
		},
		{
			name:            "Boundary - just above high risk threshold",
			anomalyScore:    0.3,
			riskProbability: 0.76,
			expected:        models.DecisionReview,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := engine.Decide(tt.anomalyScore, tt.riskProbability)
			if result != tt.expected {
				t.Errorf("Decide(%v, %v) = %v, expected %v",
					tt.anomalyScore, tt.riskProbability, result, tt.expected)
			}
		})
	}
}

func TestEngine_DecideWithExplanation(t *testing.T) {
	engine := NewEngine()

	tests := []struct {
		name            string
		anomalyScore    float64
		riskProbability float64
		expectedDec     string
	}{
		{
			name:            "High anomaly explanation",
			anomalyScore:    0.9,
			riskProbability: 0.3,
			expectedDec:     models.DecisionReview,
		},
		{
			name:            "Both high explanation",
			anomalyScore:    0.9,
			riskProbability: 0.9,
			expectedDec:     models.DecisionReview,
		},
		{
			name:            "Approve explanation",
			anomalyScore:    0.2,
			riskProbability: 0.2,
			expectedDec:     models.DecisionApprove,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			decision, explanation := engine.DecideWithExplanation(tt.anomalyScore, tt.riskProbability)
			if decision != tt.expectedDec {
				t.Errorf("DecideWithExplanation decision = %v, expected %v", decision, tt.expectedDec)
			}
			if explanation == "" {
				t.Error("Expected non-empty explanation")
			}
		})
	}
}

func TestEngine_CalculateCombinedScore(t *testing.T) {
	engine := NewEngine()

	tests := []struct {
		name            string
		anomalyScore    float64
		riskProbability float64
		expected        float64
	}{
		{
			name:            "Both zero",
			anomalyScore:    0.0,
			riskProbability: 0.0,
			expected:        0.0,
		},
		{
			name:            "Both one",
			anomalyScore:    1.0,
			riskProbability: 1.0,
			expected:        1.0,
		},
		{
			name:            "Weighted average",
			anomalyScore:    0.5,
			riskProbability: 0.5,
			expected:        0.5,
		},
		{
			name:            "Higher risk weight",
			anomalyScore:    0.0,
			riskProbability: 1.0,
			expected:        0.6, // 0*0.4 + 1*0.6
		},
		{
			name:            "Lower anomaly weight",
			anomalyScore:    1.0,
			riskProbability: 0.0,
			expected:        0.4, // 1*0.4 + 0*0.6
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := engine.CalculateCombinedScore(tt.anomalyScore, tt.riskProbability)
			if result != tt.expected {
				t.Errorf("CalculateCombinedScore(%v, %v) = %v, expected %v",
					tt.anomalyScore, tt.riskProbability, result, tt.expected)
			}
		})
	}
}

func TestNewEngineWithThresholds(t *testing.T) {
	engine := NewEngineWithThresholds(0.9, 0.85, 0.6, 0.55)

	// Test with custom thresholds
	// 0.85 should be FLAG with custom threshold (< 0.9 high anomaly)
	result := engine.Decide(0.85, 0.3)
	if result != models.DecisionFlag {
		t.Errorf("Expected FLAG with custom thresholds, got %v", result)
	}

	// 0.91 should be REVIEW with custom threshold
	result = engine.Decide(0.91, 0.3)
	if result != models.DecisionReview {
		t.Errorf("Expected REVIEW with custom thresholds, got %v", result)
	}
}
