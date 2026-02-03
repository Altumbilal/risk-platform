#!/bin/bash
# Test script for running all tests locally

set -e

echo "======================================"
echo "Running Go Tests"
echo "======================================"
cd gateway-go
go test ./... -v
cd ..

echo ""
echo "======================================"
echo "Running PyTorch Service Tests"
echo "======================================"
cd ml-anomaly-pytorch
python -m pytest tests/ -v
cd ..

echo ""
echo "======================================"
echo "Running TensorFlow Service Tests"
echo "======================================"
cd ml-risk-tensorflow
python -m pytest tests/ -v
cd ..

echo ""
echo "======================================"
echo "All tests passed!"
echo "======================================"
