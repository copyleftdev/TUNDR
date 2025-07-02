.PHONY: build run test clean lint deps

# Build variables
BINARY_NAME=tundr
VERSION?=0.1.0
BUILD_TIME=$(shell date +%s)
GIT_COMMIT=$(shell git rev-parse HEAD)
LDFLAGS=-X main.version=$(VERSION) -X main.buildTime=$(BUILD_TIME) -X main.gitCommit=$(GIT_COMMIT)

# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOTEST=$(GOCMD) test
GOGET=$(GOCMD) get
GOMOD=$(GOCMD) mod
BINARY_DIR=bin

# Build the application
build:
	@echo "Building $(BINARY_NAME)..."
	@mkdir -p $(BINARY_DIR)
	$(GOBUILD) -v -ldflags "$(LDFLAGS)" -o $(BINARY_DIR)/$(BINARY_NAME) ./cmd/server

# Run the application
run: build
	@echo "Starting $(BINARY_NAME)..."
	./$(BINARY_DIR)/$(BINARY_NAME)

# Run tests
test:
	$(GOTEST) -v ./...

# Clean build files
clean:
	@echo "Cleaning..."
	@rm -rf $(BINARY_DIR)
	@echo "Clean complete"

# Install dependencies
deps:
	$(GOMOD) download

# Lint the code
lint:
	@echo "Linting..."
	golangci-lint run ./...

# Run with hot reload for development
dev:
	@echo "Starting in development mode..."
	nodemon --watch . --ext go --exec "make run" --signal SIGTERM

# Generate Swagger docs
swagger:
	swag init -g cmd/server/main.go

# Run migrations
migrate-up:
	@echo "Running migrations..."
	migrate -path migrations -database "${DB_DSN}" up

migrate-down:
	@echo "Reverting migrations..."
	migrate -path migrations -database "${DB_DSN}" down 1

# Help
help:
	@echo "\nAvailable commands:"
	@echo "  make build    - Build the application"
	@echo "  make run      - Run the application"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Clean build files"
	@echo "  make deps     - Install dependencies"
	@echo "  make lint     - Lint the code"
	@echo "  make dev      - Run with hot reload"
	@echo "  make swagger  - Generate Swagger docs"
	@echo "  make migrate-up   - Run database migrations"
	@echo "  make migrate-down - Rollback last migration"

.DEFAULT_GOAL := build
