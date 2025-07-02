package errors

import (
	"net/http"
	"runtime/debug"

	"github.com/copyleftdev/TUNDR/internal/logging"
)

// RecoveryMiddleware returns a middleware that recovers from panics.
func RecoveryMiddleware(logger *logging.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if rec := recover(); rec != nil {
					// Create a map to hold panic info
					fields := map[string]interface{}{
						"error": rec,
						"stack": string(debug.Stack()),
					}

					// Add request info if available
					if r != nil {
						fields["method"] = r.Method
						fields["path"] = r.URL.Path
						fields["query"] = r.URL.RawQuery
						fields["headers"] = r.Header
					}

					// Log the panic
					logger.Error("Recovered from panic", fields)


					// Return a 500 Internal Server Error
					http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
				}
			}()

			next.ServeHTTP(w, r)
		})
	}
}

// ErrorHandler is a middleware that handles errors returned by HTTP handlers.
func ErrorHandler(logger *logging.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Create a response writer that captures the status code
			rw := &responseWriter{ResponseWriter: w, status: http.StatusOK}

			// Call the next handler
			next.ServeHTTP(rw, r)


			// Log errors (status >= 400)
			if rw.status >= http.StatusBadRequest {
				logger.Error("Request error", map[string]interface{}{
					"status":  rw.status,
					"method":  r.Method,
					"path":    r.URL.Path,
					"query":   r.URL.RawQuery,
					"ip":      r.RemoteAddr,
				})
			}
		})
	}
}

// responseWriter wraps http.ResponseWriter to capture the status code.
type responseWriter struct {
	http.ResponseWriter
	status int
}

// WriteHeader captures the status code before writing the header.
func (rw *responseWriter) WriteHeader(code int) {
	rw.status = code
	rw.ResponseWriter.WriteHeader(code)
}
