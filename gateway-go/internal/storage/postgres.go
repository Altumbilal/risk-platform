package storage

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	_ "github.com/lib/pq"
	"github.com/risk-platform/gateway-go/internal/models"
)

// PostgresDB handles PostgreSQL database operations
type PostgresDB struct {
	db *sql.DB
}

// NewPostgresDB creates a new PostgreSQL connection
func NewPostgresDB(connectionURL string) (*PostgresDB, error) {
	db, err := sql.Open("postgres", connectionURL)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Configure connection pool
	db.SetMaxOpenConns(25)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(5 * time.Minute)

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	pg := &PostgresDB{db: db}

	// Initialize schema
	if err := pg.initSchema(ctx); err != nil {
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	return pg, nil
}

// initSchema creates the required tables
func (p *PostgresDB) initSchema(ctx context.Context) error {
	schema := `
		CREATE TABLE IF NOT EXISTS transactions (
			id UUID PRIMARY KEY,
			transaction_id VARCHAR(255) NOT NULL UNIQUE,
			payload JSONB NOT NULL,
			anomaly_score FLOAT NOT NULL,
			risk_score FLOAT NOT NULL,
			decision VARCHAR(50) NOT NULL,
			created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
		);

		CREATE INDEX IF NOT EXISTS idx_transactions_transaction_id ON transactions(transaction_id);
		CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at);
		CREATE INDEX IF NOT EXISTS idx_transactions_decision ON transactions(decision);
	`

	_, err := p.db.ExecContext(ctx, schema)
	return err
}

// SaveTransaction persists a transaction record
func (p *PostgresDB) SaveTransaction(ctx context.Context, record *models.TransactionRecord) error {
	query := `
		INSERT INTO transactions (id, transaction_id, payload, anomaly_score, risk_score, decision, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		ON CONFLICT (transaction_id) DO UPDATE SET
			anomaly_score = EXCLUDED.anomaly_score,
			risk_score = EXCLUDED.risk_score,
			decision = EXCLUDED.decision
	`

	_, err := p.db.ExecContext(ctx, query,
		record.ID,
		record.TransactionID,
		record.Payload,
		record.AnomalyScore,
		record.RiskScore,
		record.Decision,
		record.CreatedAt,
	)

	return err
}

// GetTransaction retrieves a transaction by ID
func (p *PostgresDB) GetTransaction(ctx context.Context, transactionID string) (*models.TransactionRecord, error) {
	query := `
		SELECT id, transaction_id, payload, anomaly_score, risk_score, decision, created_at
		FROM transactions
		WHERE transaction_id = $1
	`

	var record models.TransactionRecord
	err := p.db.QueryRowContext(ctx, query, transactionID).Scan(
		&record.ID,
		&record.TransactionID,
		&record.Payload,
		&record.AnomalyScore,
		&record.RiskScore,
		&record.Decision,
		&record.CreatedAt,
	)

	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get transaction: %w", err)
	}

	return &record, nil
}

// GetTransactionsByDecision retrieves transactions by decision type
func (p *PostgresDB) GetTransactionsByDecision(ctx context.Context, decision string, limit int) ([]*models.TransactionRecord, error) {
	query := `
		SELECT id, transaction_id, payload, anomaly_score, risk_score, decision, created_at
		FROM transactions
		WHERE decision = $1
		ORDER BY created_at DESC
		LIMIT $2
	`

	rows, err := p.db.QueryContext(ctx, query, decision, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query transactions: %w", err)
	}
	defer rows.Close()

	var records []*models.TransactionRecord
	for rows.Next() {
		var record models.TransactionRecord
		if err := rows.Scan(
			&record.ID,
			&record.TransactionID,
			&record.Payload,
			&record.AnomalyScore,
			&record.RiskScore,
			&record.Decision,
			&record.CreatedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan row: %w", err)
		}
		records = append(records, &record)
	}

	return records, nil
}

// Close closes the database connection
func (p *PostgresDB) Close() error {
	return p.db.Close()
}

// Health checks database connectivity
func (p *PostgresDB) Health(ctx context.Context) error {
	return p.db.PingContext(ctx)
}
