package decision

import (
	"github.com/risk-platform/gateway-go/internal/models"
)

// Engine implements decision logic based on ML outputs
type Engine struct {
	// Thresholds for decision making
	HighAnomalyThreshold float64
	HighRiskThreshold    float64
	MedAnomalyThreshold  float64
	MedRiskThreshold     float64
}

// NewEngine creates a new decision engine with default thresholds
func NewEngine() *Engine {
	return &Engine{
		HighAnomalyThreshold: 0.8,
		HighRiskThreshold:    0.75,
		MedAnomalyThreshold:  0.5,
		MedRiskThreshold:     0.5,
	}
}

// NewEngineWithThresholds creates an engine with custom thresholds
func NewEngineWithThresholds(highAnomaly, highRisk, medAnomaly, medRisk float64) *Engine {
	return &Engine{
		HighAnomalyThreshold: highAnomaly,
		HighRiskThreshold:    highRisk,
		MedAnomalyThreshold:  medAnomaly,
		MedRiskThreshold:     medRisk,
	}
}

// Decide makes a decision based on anomaly score and risk probability
// Decision logic:
//   - REVIEW: anomaly_score > 0.8 OR risk_probability > 0.75
//   - FLAG: anomaly_score > 0.5 OR risk_probability > 0.5
//   - APPROVE: otherwise
func (e *Engine) Decide(anomalyScore, riskProbability float64) string {
	// High risk - requires manual review
	if anomalyScore > e.HighAnomalyThreshold || riskProbability > e.HighRiskThreshold {
		return models.DecisionReview
	}

	// Medium risk - flag for monitoring
	if anomalyScore > e.MedAnomalyThreshold || riskProbability > e.MedRiskThreshold {
		return models.DecisionFlag
	}

	// Low risk - approve
	return models.DecisionApprove
}

// DecideWithExplanation returns decision with reasoning
func (e *Engine) DecideWithExplanation(anomalyScore, riskProbability float64) (string, string) {
	decision := e.Decide(anomalyScore, riskProbability)

	var explanation string
	switch decision {
	case models.DecisionReview:
		if anomalyScore > e.HighAnomalyThreshold && riskProbability > e.HighRiskThreshold {
			explanation = "Both anomaly score and risk probability exceed high thresholds"
		} else if anomalyScore > e.HighAnomalyThreshold {
			explanation = "Anomaly score exceeds high threshold"
		} else {
			explanation = "Risk probability exceeds high threshold"
		}
	case models.DecisionFlag:
		if anomalyScore > e.MedAnomalyThreshold && riskProbability > e.MedRiskThreshold {
			explanation = "Both anomaly score and risk probability exceed medium thresholds"
		} else if anomalyScore > e.MedAnomalyThreshold {
			explanation = "Anomaly score exceeds medium threshold"
		} else {
			explanation = "Risk probability exceeds medium threshold"
		}
	case models.DecisionApprove:
		explanation = "All risk indicators within acceptable thresholds"
	}

	return decision, explanation
}

// CalculateCombinedScore returns a weighted combined score
func (e *Engine) CalculateCombinedScore(anomalyScore, riskProbability float64) float64 {
	// Weighted average: 40% anomaly, 60% risk classification
	return (anomalyScore * 0.4) + (riskProbability * 0.6)
}
