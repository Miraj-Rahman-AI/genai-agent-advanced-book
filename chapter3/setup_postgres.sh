#!/bin/bash

# ============================================================
# PostgreSQL Setup Script
# ------------------------------------------------------------
# This script sets up a PostgreSQL database environment using
# Docker Compose. It performs the following:
#   - Checks for required dependencies (Docker & Docker Compose)
#   - Validates required configuration files
#   - Stops existing containers if running
#   - Starts a new PostgreSQL container
#   - Waits for the database to be ready
#   - Verifies database initialization and connectivity
# ============================================================

set -e  # Stop execution immediately if any command fails

# ============================================================
# Helper functions for colored terminal output
# These improve readability by categorizing log messages.
# ============================================================
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# ============================================================
# Determine the directory where this script is located
# and switch execution to that directory.
# This ensures relative paths (docker-compose.yml, init.sql)
# work correctly regardless of where the script is run from.
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_info "Starting PostgreSQL setup..."

# ============================================================
# Check if Docker is installed
# Docker is required to run the PostgreSQL container.
# ============================================================
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed."
    print_info "Please install Docker using one of the following commands:"
    print_info "  Ubuntu/Debian: sudo apt update && sudo apt install docker.io docker-compose"
    print_info "  macOS: brew install docker docker-compose"
    exit 1
fi

# ============================================================
# Check if Docker Compose is installed
# Docker Compose is required to orchestrate the PostgreSQL service.
# ============================================================
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed."
    print_info "Install Docker Compose using:"
    print_info "  Ubuntu/Debian: sudo apt install docker-compose"
    print_info "  macOS: brew install docker-compose"
    exit 1
fi

# ============================================================
# Verify required configuration files exist
# ============================================================
if [ ! -f "docker-compose.yml" ]; then
    print_error "docker-compose.yml file not found."
    exit 1
fi

if [ ! -f "init.sql" ]; then
    print_error "init.sql file not found."
    exit 1
fi

print_success "All required configuration files are present."

# ============================================================
# Stop and remove any existing PostgreSQL containers
# This prevents conflicts with previously running instances.
# The '-v' option removes volumes to ensure a clean database.
# ============================================================
print_info "Stopping and removing existing PostgreSQL containers..."
docker-compose down -v 2>/dev/null || true

# ============================================================
# Check whether port 5432 is already in use
# This is the default PostgreSQL port and may conflict with
# local installations or other running containers.
# ============================================================
print_info "Checking port 5432 usage..."
if netstat -tlnp 2>/dev/null | grep -q :5432; then
    print_warning "Port 5432 is already in use. This may cause conflicts."
    print_info "Please stop any existing PostgreSQL services:"
    print_info "  System service: sudo service postgresql stop"
    print_info "  Or: sudo systemctl stop postgresql"
    
    # Prompt user confirmation before continuing
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Setup aborted by user."
        exit 1
    fi
fi

# ============================================================
# Start PostgreSQL container using Docker Compose
# ============================================================
print_info "Starting PostgreSQL container..."
docker-compose up -d

# ============================================================
# Wait for PostgreSQL container to initialize
# Initial wait allows container to boot.
# ============================================================
print_info "Waiting for PostgreSQL to initialize..."
sleep 5

# Maximum wait time (seconds)
TIMEOUT=30
COUNTER=0

# Poll until PostgreSQL is ready or timeout reached
while [ $COUNTER -lt $TIMEOUT ]; do
    if docker exec postgres-genai-ch3 pg_isready -U testuser -d testdb &>/dev/null; then
        print_success "PostgreSQL started successfully!"
        break
    fi
    
    print_info "Waiting for PostgreSQL startup... ($COUNTER/$TIMEOUT seconds)"
    sleep 1
    COUNTER=$((COUNTER + 1))
done

# If timeout reached without success, show error and logs
if [ $COUNTER -eq $TIMEOUT ]; then
    print_error "PostgreSQL failed to start within expected time."
    print_info "Check logs using:"
    print_info "  docker-compose logs postgres"
    exit 1
fi

# ============================================================
# Verify database initialization
# Check if the employees table exists and contains data.
# ============================================================
print_info "Verifying database initialization..."
if docker exec postgres-genai-ch3 psql -U testuser -d testdb -c "SELECT COUNT(*) FROM employees;" &>/dev/null; then
    employee_count=$(docker exec postgres-genai-ch3 psql -U testuser -d testdb -t -c "SELECT COUNT(*) FROM employees;" | xargs)
    print_success "Employees table created successfully. Record count: $employee_count"
else
    print_error "Database initialization failed."
    exit 1
fi

# ============================================================
# Test database connectivity
# ============================================================
print_info "Running database connection test..."
if docker exec postgres-genai-ch3 psql -U testuser -d testdb -c "SELECT version();" &>/dev/null; then
    print_success "Database connection test succeeded!"
else
    print_error "Database connection test failed."
    exit 1
fi

# ============================================================
# Final setup summary
# ============================================================
print_success "PostgreSQL setup completed successfully!"
echo
print_info "=== Setup Information ==="
print_info "Container Name: postgres-genai-ch3"
print_info "Database: testdb"
print_info "User: testuser"
print_info "Password: testpass"
print_info "Port: 5432"
print_info "Host: localhost"
echo

# ============================================================
# Useful management commands
# ============================================================
print_info "=== Available Commands ==="
print_info "Connect to database:"
print_info "  docker exec -it postgres-genai-ch3 psql -U testuser -d testdb"
echo
print_info "View employees table:"
print_info "  docker exec postgres-genai-ch3 psql -U testuser -d testdb -c \"SELECT * FROM employees;\""
echo
print_info "Stop container:"
print_info "  docker-compose down"
echo
print_info "Restart container:"
print_info "  docker-compose up -d"
echo

# ============================================================
# Instructions for use in Jupyter Notebook environment
# ============================================================
print_info "=== Usage in Jupyter Notebook ==="
print_info "1. Set your OpenAI API key in the .env file"
print_info "2. Open notebooks/examples.ipynb"
print_info "3. Run the SQLDatabaseChain usage section"
echo

print_success "Setup completed successfully."
