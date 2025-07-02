package logging

import (
	"context"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5/middleware"
)

// Middleware returns a middleware that logs the start and end of each request.
func Middleware(logger *Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()

			// Create a response writer wrapper to capture the status code
			ww := middleware.NewWrapResponseWriter(w, r.ProtoMajor)

			// Add logger to context
			reqID := middleware.GetReqID(r.Context())
			requestLogger := logger.WithFields(map[string]interface{}{
				"request_id": reqID,
				"method":     r.Method,
				"path":       r.URL.Path,
				"remote":     r.RemoteAddr,
			})

			// Log request start
			requestLogger.Info("Request started")


			// Create context with logger
			ctx := context.WithValue(r.Context(), ctxLoggerKey{}, &CtxLogger{requestLogger})

			// Process the request
			next.ServeHTTP(ww, r.WithContext(ctx))


			// Calculate response time
			latency := time.Since(start)


			// Log request completion
			fields := map[string]interface{}{
				"status":      ww.Status(),
				"bytes":       ww.BytesWritten(),
				"latency_ms":  float64(latency.Microseconds()) / 1000.0,
				"user_agent":  r.UserAgent(),
				"protocol":    r.Proto,
				"latency":     latency.String(),
			}

			// Add error if present
			if ww.Status() >= 400 {
				fields["error"] = http.StatusText(ww.Status())
			}

			requestLogger.WithFields(fields).Info("Request completed")
		})
	}
}
