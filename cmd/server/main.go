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

	// Initialize base logger
	logger, err := logging.NewLogger(&logging.Config{
		Level:  cfg.Logging.Level,
		Format: cfg.Logging.Format,
		Output: cfg.Logging.Output,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize logger: %v\n", err)
		os.Exit(1)
	}

	// Create a context with logger
	ctx := context.Background()
	
	// Create a service logger with additional fields
	serviceLogger := logger.WithFields(map[string]interface{}{
		"service": "mcp-optimization-server",
		"version": "1.0.0",
	})

	// Create a context logger
	ctxLogger := &logging.CtxLogger{Logger: serviceLogger}
	ctx = ctxLogger.WithContext(ctx)

	// Create router
	r := chi.NewRouter()

	// Add middleware
	r.Use(middleware.RequestID)
	r.Use(middleware.RealIP)
	r.Use(logging.Middleware(logger)) // Our custom logging middleware
	
	// Error handling and recovery
	r.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if err := recover(); err != nil {
					serviceLogger.Error("Recovered from panic", map[string]interface{}{"error": fmt.Errorf("%v", err)})
					http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
				}
			}()
			next.ServeHTTP(w, r)
		})
	})
	r.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			next.ServeHTTP(w, r)
			if err := r.Context().Err(); err != nil {
				serviceLogger.Error("Request error", map[string]interface{}{"error": fmt.Errorf("%v", err)})
				http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
			}
		})
	})
	
	// Timeout and other standard middleware
	r.Use(middleware.Timeout(60 * time.Second))

	// Add request context logger
	r.Use(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Get the logger from context or use the base logger
			ctxLogger := logging.FromContext(r.Context())
			if ctxLogger == nil {
				ctxLogger = &logging.CtxLogger{Logger: logger}
			}

			// Add request ID to logger
			reqLogger := ctxLogger.Logger.WithFields(map[string]interface{}{
				"request_id": middleware.GetReqID(r.Context()),
			})

			// Create a new context with the request logger
			reqCtxLogger := &logging.CtxLogger{Logger: reqLogger}
			reqCtx := reqCtxLogger.WithContext(r.Context())
			next.ServeHTTP(w, r.WithContext(reqCtx))
		})
	})

	// Add health check endpoint
	r.Get("/healthz", func(w http.ResponseWriter, r *http.Request) {
		logger := logging.FromContext(r.Context())
		if logger != nil {
			logger.Info("Health check")
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	// Add metrics endpoint
	r.Handle("/metrics", promhttp.Handler())

	// Create server instance with our logger
	srv := server.NewServer(cfg, serviceLogger)
	srv.RegisterRoutes(r)

	// Start server
	httpServer := &http.Server{
		Addr:    fmt.Sprintf(":%d", cfg.HTTP.Port),
		Handler: r,
	}

	// Start HTTP server
	go func() {
		serviceLogger.Info("Starting server", map[string]interface{}{
			"address": httpServer.Addr,
		})

		if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			serviceLogger.Fatal("Failed to start server", map[string]interface{}{
				"error": err.Error(),
			})
		}
	}()

	// Wait for interrupt signal to gracefully shut down the server
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	serviceLogger.Info("Shutting down server...")

	// Create a deadline to wait for
	shutdownCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	// Shutdown the server
	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		serviceLogger.Error("Server forced to shutdown", map[string]interface{}{"error": err})
		os.Exit(1)
	}

	serviceLogger.Info("Server stopped")

	if err := srv.Close(); err != nil {
		serviceLogger.Error("error closing server resources", map[string]interface{}{"error": err})
	}

	serviceLogger.Info("server exited properly")
}
