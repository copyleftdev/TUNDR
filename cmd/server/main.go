package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/copyleftdev/TUNDR/internal/config"
	"github.com/copyleftdev/TUNDR/internal/errors"
	"github.com/copyleftdev/TUNDR/internal/logging"
	"github.com/copyleftdev/TUNDR/internal/server"
)

func main() {
	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		// Use standard logger as fallback if config loading fails
		fmt.Fprintf(os.Stderr, "Failed to load configuration: %v\n", err)
		os.Exit(1)
	}

	// Initialize logger
	logger, err := logging.NewLogger(&logging.Config{
		Level:  cfg.LogLevel,
		Format: cfg.LogFormat,
		Output: cfg.LogOutput,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize logger: %v\n", err)
		os.Exit(1)
	}

	// Create a context with logger
	ctx := context.Background()
	ctxLogger := logging.FromContext(ctx).WithFields(map[string]interface{}{
		"service": "mcp-optimization-server",
		"version": "1.0.0",
	})
	ctx = ctxLogger.WithContext(ctx)

	// Create router
	r := chi.NewRouter()

	// Add middleware
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(logging.Middleware(logger)) // Our custom logging middleware
	
	// Error handling and recovery
	r.Use(errors.RecoveryMiddleware(logger)) // Custom panic recovery
	r.Use(errors.ErrorHandler(logger))       // Error response handling
	
	// Timeout and other standard middleware
	r.Use(middleware.Timeout(60 * time.Second))

	// Add request context logger
	r.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Add logger to request context
			reqLogger := logger.WithFields(map[string]interface{}{
				"request_id": middleware.GetReqID(r.Context()),
			})
			ctx := reqLogger.WithContext(r.Context())
			next.ServeHTTP(w, r.WithContext(ctx))
		})
	})

	// Add health check endpoint
	r.Get("/healthz", func(w http.ResponseWriter, r *http.Request) {
		logger := logging.FromContext(r.Context())
		logger.Info("Health check")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	// Add metrics endpoint
	r.Handle("/metrics", promhttp.Handler())

	// Initialize and register API routes
	srv := server.NewServer(cfg, logger)
	srv.RegisterRoutes(r)

	// Start server
	srvHTTP := &http.Server{
		Addr:    fmt.Sprintf(":%d", cfg.HTTP.Port),
		Handler: r,
	}

	// Add metrics endpoint
	r.Handle("/metrics", promhttp.Handler())

	// Start server in a goroutine
	go func() {
		logger.Info("Starting server", map[string]interface{}{"port": cfg.HTTP.Port})
		if err := srvHTTP.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logger.Error("Failed to start server", map[string]interface{}{"error": err})
			os.Exit(1)
		}
	}()

	// Wait for interrupt signal to gracefully shut down the server
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("Shutting down server...")

	// Create a deadline to wait for
	shutdownCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	// Shutdown the server
	if err := srvHTTP.Shutdown(shutdownCtx); err != nil {
		logger.Error("Server forced to shutdown", map[string]interface{}{"error": err})
		os.Exit(1)
	}

	logger.Info("Server exited gracefully")

	if err := srv.Close(); err != nil {
		logger.Error("error closing server resources", map[string]interface{}{"error": err})
	}

	logger.Info("server exited properly")
}
