-- Initialize database schema for risk platform

-- Create transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY,
    transaction_id VARCHAR(255) NOT NULL UNIQUE,
    payload JSONB NOT NULL,
    anomaly_score FLOAT NOT NULL,
    risk_score FLOAT NOT NULL,
    decision VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_transactions_transaction_id ON transactions (transaction_id);

CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions (created_at);

CREATE INDEX IF NOT EXISTS idx_transactions_decision ON transactions (decision);

CREATE INDEX IF NOT EXISTS idx_transactions_risk_score ON transactions (risk_score);

-- Create a view for high-risk transactions
CREATE OR REPLACE VIEW high_risk_transactions AS
SELECT
    id,
    transaction_id,
    payload,
    anomaly_score,
    risk_score,
    decision,
    created_at
FROM transactions
WHERE
    decision IN ('REVIEW', 'DENY')
ORDER BY created_at DESC;

-- Create a function to get transaction statistics
CREATE OR REPLACE FUNCTION get_transaction_stats()
RETURNS TABLE (
    total_count BIGINT,
    approve_count BIGINT,
    flag_count BIGINT,
    review_count BIGINT,
    deny_count BIGINT,
    avg_anomaly_score FLOAT,
    avg_risk_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT as total_count,
        COUNT(*) FILTER (WHERE decision = 'APPROVE')::BIGINT as approve_count,
        COUNT(*) FILTER (WHERE decision = 'FLAG')::BIGINT as flag_count,
        COUNT(*) FILTER (WHERE decision = 'REVIEW')::BIGINT as review_count,
        COUNT(*) FILTER (WHERE decision = 'DENY')::BIGINT as deny_count,
        AVG(anomaly_score)::FLOAT as avg_anomaly_score,
        AVG(risk_score)::FLOAT as avg_risk_score
    FROM transactions;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;