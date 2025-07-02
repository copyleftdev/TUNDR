package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/copyleftdev/TUNDR/internal/config"
	"github.com/copyleftdev/TUNDR/internal/logging"
	"github.com/go-chi/chi/v5"
	"github.com/stretchr/testify/assert"
)

// testConfig creates a test configuration with default values
func testConfig(t *testing.T) *config.Config {
	cfg := &config.Config{
		Environment: "test",
	}

	// Set up HTTP config
	cfg.HTTP.Port = 8080
	cfg.HTTP.ReadTimeout = 30 * time.Second
	cfg.HTTP.WriteTimeout = 30 * time.Second
	cfg.HTTP.IdleTimeout = 120 * time.Second
	cfg.HTTP.ShutdownTimeout = 30 * time.Second

	// Set up logging
	cfg.Logging.Level = "debug"
	cfg.Logging.Format = "console"
	cfg.Logging.Output = "stdout"

	// Set up database
	cfg.Database.Type = "sqlite"
	cfg.Database.DSN = "file::memory:?cache=shared"
	cfg.Database.MaxConns = 5

	// Set up auth
	cfg.Auth.Enabled = false
	cfg.Auth.JWTKey = "test-key"

	// Set up optimization
	cfg.Optimization.WorkerCount = 3

	return cfg
}

// testLogger creates a test logger
func testLogger(t *testing.T) *logging.Logger {
	logger, err := logging.NewLogger(&logging.Config{
		Level:  "debug",
		Format: "console",
		Output: "stdout",
	})
	if err != nil {
		t.Fatalf("Failed to create logger: %v", err)
	}
	return logger
}

func TestNewServer(t *testing.T) {
	// Create a test logger and config
	logger := testLogger(t)
	cfg := testConfig(t)

	// Test server creation
	srv := NewServer(cfg, logger)
	assert.NotNil(t, srv, "Server should be created")
}

func TestRegisterRoutes(t *testing.T) {
	// Create a test logger and config
	logger := testLogger(t)
	cfg := testConfig(t)

	// Create server and register routes
	srv := NewServer(cfg, logger)
	r := chi.NewRouter()
	srv.RegisterRoutes(r)

	// Test if routes are registered
	tests := []struct {
		method string
		path   string
		shouldExist bool
	}{
		{"POST", "/api/v1/optimize", true},
		{"GET", "/api/v1/status/123", true},
		{"DELETE", "/api/v1/optimization/123", true},
		{"POST", "/rpc", true},
		{"GET", "/healthz", false},  // Not registered by server package
		{"GET", "/nonexistent", false},  // Should not exist
	}

	for _, tt := range tests {
		t.Run(tt.method+" "+tt.path, func(t *testing.T) {
			req := httptest.NewRequest(tt.method, tt.path, nil)
			rr := httptest.NewRecorder()
			r.ServeHTTP(rr, req)
			
			// We're just checking if the route exists, not the response
			// A 404 would mean the route doesn't exist
			if tt.shouldExist && rr.Code == http.StatusNotFound {
				t.Errorf("Route %s %s should exist but returned 404", tt.method, tt.path)
			}
		})
	}
}

func TestClose(t *testing.T) {
	// Create a test logger and config
	logger := testLogger(t)
	cfg := testConfig(t)

	// Test server close
	srv := NewServer(cfg, logger)
	err := srv.Close()
	assert.NoError(t, err, "Close should not return an error")
}

func TestRespondWithError(t *testing.T) {
	// Create a test logger and config
	logger := testLogger(t)
	cfg := testConfig(t)

	srv := NewServer(cfg, logger)

	tests := []struct {
		name       string
		code       int
		message    string
		id         interface{}
		expectedID  interface{}
		expectCode  int
		expectError bool
	}{
		{
			name:      "valid error response",
			code:      http.StatusBadRequest,
			message:   "invalid input",
			id:        "123",
			expectedID: "123",
			expectCode: http.StatusOK, // Because respondWithError writes 200 with error in body
		},
		{
			name:      "nil id",
			code:      http.StatusInternalServerError,
			message:   "server error",
			id:        nil,
			expectedID: nil,
			expectCode: http.StatusOK,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rr := httptest.NewRecorder()
			srv.respondWithError(rr, tt.code, tt.message, tt.id)

			assert.Equal(t, tt.expectCode, rr.Code, "status code should match")

			// Parse response body to verify error structure
			var response map[string]interface{}
			err := json.NewDecoder(rr.Body).Decode(&response)
			assert.NoError(t, err, "should decode response body")

			// Check error object
			errObj, ok := response["error"].(map[string]interface{})
			assert.True(t, ok, "response should contain error object")
			assert.Equal(t, float64(tt.code), errObj["code"], "error code should match")
			assert.Equal(t, tt.message, errObj["message"], "error message should match")

			// Check ID
			assert.Equal(t, tt.expectedID, response["id"], "response ID should match")
		})
	}
}
