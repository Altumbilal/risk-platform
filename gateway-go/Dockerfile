# Build stage
FROM golang:1.22-alpine AS builder

WORKDIR /app

# Install dependencies (update index first to avoid network issues)
RUN apk update && apk add --no-cache git

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o gateway ./cmd/main.go

# Final stage
FROM alpine:3.19

WORKDIR /app

# Add ca-certificates for HTTPS
RUN apk --no-cache add ca-certificates tzdata

# Copy binary from builder
COPY --from=builder /app/gateway .

# Expose port
EXPOSE 8080

# Run the application
CMD ["./gateway"]
