#!/bin/bash
# DIGIMON Test Runner Script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🧪 DIGIMON Test Runner"
echo "====================="

# Parse command line arguments
RUN_EXPENSIVE=false
COVERAGE=false
PARALLEL=false
SPECIFIC_TEST=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --expensive)
            RUN_EXPENSIVE=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --test)
            SPECIFIC_TEST="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--expensive] [--coverage] [--parallel] [--test <test_name>]"
            exit 1
            ;;
    esac
done

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}❌ pytest is not installed. Installing...${NC}"
    pip install pytest pytest-asyncio pytest-cov pytest-timeout pytest-mock pytest-xdist
fi

# Build pytest command
PYTEST_CMD="pytest -v"

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=Core --cov-report=html --cov-report=term-missing"
fi

if [ "$PARALLEL" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -n auto"
fi

if [ "$RUN_EXPENSIVE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --run-expensive"
fi

if [ -n "$SPECIFIC_TEST" ]; then
    PYTEST_CMD="$PYTEST_CMD $SPECIFIC_TEST"
fi

# Run linting first
echo -e "\n${YELLOW}🔍 Running code quality checks...${NC}"
echo "================================="

# Check if black is installed
if command -v black &> /dev/null; then
    echo "Running Black formatter check..."
    black --check Core/ tests/ 2>/dev/null || echo -e "${YELLOW}⚠️  Some files need formatting${NC}"
fi

# Check if flake8 is installed
if command -v flake8 &> /dev/null; then
    echo "Running Flake8 linter..."
    flake8 Core/ tests/ 2>/dev/null || echo -e "${YELLOW}⚠️  Some linting issues found${NC}"
fi

# Run tests
echo -e "\n${GREEN}🚀 Running tests...${NC}"
echo "==================="
echo "Command: $PYTEST_CMD"
echo ""

# Set environment variables for testing
export DIGIMON_LOG_LEVEL="INFO"
export OPENAI_API_KEY="test-key"

# Run pytest
$PYTEST_CMD

# Check exit code
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ All tests passed!${NC}"
    
    if [ "$COVERAGE" = true ]; then
        echo -e "\n${GREEN}📊 Coverage report generated in htmlcov/index.html${NC}"
    fi
else
    echo -e "\n${RED}❌ Some tests failed!${NC}"
    exit 1
fi