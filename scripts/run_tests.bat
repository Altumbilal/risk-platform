@echo off
REM Test script for running all tests locally on Windows

echo ======================================
echo Running Go Tests
echo ======================================
cd gateway-go
go test ./... -v
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
cd ..

echo.
echo ======================================
echo Running PyTorch Service Tests
echo ======================================
cd ml-anomaly-pytorch
python -m pytest tests/ -v
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
cd ..

echo.
echo ======================================
echo Running TensorFlow Service Tests
echo ======================================
cd ml-risk-tensorflow
python -m pytest tests/ -v
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%
cd ..

echo.
echo ======================================
echo All tests passed!
echo ======================================
