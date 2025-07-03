package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/copyleftdev/TUNDR/internal/config"
	"github.com/copyleftdev/TUNDR/internal/logging"
	"github.com/copyleftdev/TUNDR/internal/optimization"
	"github.com/copyleftdev/TUNDR/internal/optimization/bayesian"
)

// Logger defines the logging interface used by the server
// This allows us to be flexible with our logging implementation
type Logger interface {
	Debug(msg string, fields ...map[string]interface{})
	Info(msg string, fields ...map[string]interface{})
	Warn(msg string, fields ...map[string]interface{})
	Error(msg string, fields ...map[string]interface{})
	Fatal(msg string, fields ...map[string]interface{})
	WithFields(fields map[string]interface{}) *logging.Logger
}

// OptimizationState represents the state of an optimization job.
// It tracks the progress, status, and results of an optimization process.
// The state is thread-safe and can be accessed concurrently.
type OptimizationState struct {
	ID            string
	Status        string // "pending", "running", "completed", "failed", "cancelled"
	StartTime     time.Time
	EndTime       *time.Time
	Progress      float64
	BestSolution  *optimization.Solution
	Optimizer     optimization.Optimizer
	CancelFunc    context.CancelFunc
	LastUpdated   time.Time
}

// Server implements the HTTP and JSON-RPC server for the optimization service.
// It manages optimization jobs and provides endpoints to start, monitor, and cancel them.
type Server struct {
	cfg    *config.Config
	logger Logger
	
	// Optimization state management
	optimizations   map[string]*OptimizationState
	optimizationsMu sync.RWMutex // Protects the optimizations map
}

// NewServer creates a new server instance with the given config and logger
// The logger parameter accepts any type that implements the Logger interface
func NewServer(cfg *config.Config, logger Logger) *Server {
	return &Server{
		cfg:          cfg,
		logger:       logger,
		optimizations: make(map[string]*OptimizationState),
	}
}

func (s *Server) RegisterRoutes(r chi.Router) {
	// API v1 routes
	r.Route("/api/v1", func(r chi.Router) {
		r.Post("/optimize", s.handleOptimize)
		r.Get("/status/{id}", s.handleStatus)
		r.Delete("/optimization/{id}", s.handleCancel)
	})

	// MCP JSON-RPC 2.0 endpoint
	r.Post("/rpc", s.handleJSONRPC)
}

// handleJSONRPC handles JSON-RPC 2.0 requests
func (s *Server) handleJSONRPC(w http.ResponseWriter, r *http.Request) {
	var request struct {
		JSONRPC string        `json:"jsonrpc"`
		ID      interface{}   `json:"id"`
		Method  string        `json:"method"`
		Params  []interface{} `json:"params,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		s.respondWithError(w, -32700, "Parse error", nil)
		return
	}

	// Validate JSON-RPC 2.0 request
	if request.JSONRPC != "2.0" {
		s.respondWithError(w, -32600, "Invalid Request", nil)
		return
	}

	// Route to appropriate handler
	var result interface{}
	var err error

	switch request.Method {
	case "optimization.start":
		result, err = s.handleOptimizeStart(request.Params)
	case "optimization.status":
		result, err = s.handleOptimizationStatus(request.Params)
	case "optimization.cancel":
		err = s.handleOptimizationCancel(request.Params)
	default:
		s.respondWithError(w, -32601, "Method not found", request.ID)
		return
	}

	if err != nil {
		s.respondWithError(w, -32000, "Server error", request.ID)
		return
	}

	// Send successful response
	response := map[string]interface{}{
		"jsonrpc": "2.0",
		"id":      request.ID,
		"result":  result,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleOptimizeStart handles the optimization.start JSON-RPC method.
// It starts a new optimization job with the specified parameters.
// Expected parameters: {"objective": "function(x) { return x[0]*x[0]; }", "bounds": [[-5, 5]]}
// Returns: {"optimization_id": "opt_123", "status": "pending"}
func (s *Server) handleOptimizeStart(params []interface{}) (interface{}, error) {
	if len(params) == 0 {
		return nil, fmt.Errorf("missing required parameters")
	}

	// Parse optimization parameters
	paramMap, ok := params[0].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid parameter format, expected object")
	}

	// Extract objective function
	objectiveFn, ok := paramMap["objective"].(string)
	if !ok || objectiveFn == "" {
		return nil, fmt.Errorf("objective function is required")
	}

	// Extract bounds
	boundsInterface, ok := paramMap["bounds"].([]interface{})
	if !ok || len(boundsInterface) == 0 {
		return nil, fmt.Errorf("bounds are required")
	}

	// Convert bounds to [][2]float64
	bounds := make([][2]float64, len(boundsInterface))
	for i, b := range boundsInterface {
		bound, ok := b.([]interface{})
		if !ok || len(bound) != 2 {
			return nil, fmt.Errorf("invalid bounds format, expected [[min1, max1], [min2, max2], ...]")
		}
		min, ok1 := bound[0].(float64)
		max, ok2 := bound[1].(float64)
		if !ok1 || !ok2 {
			return nil, fmt.Errorf("bounds must be numbers")
		}
		bounds[i] = [2]float64{min, max}
	}

	// Generate a unique ID for this optimization
	id := fmt.Sprintf("opt_%d", time.Now().UnixNano())

	// Create optimization config
	config := optimization.OptimizerConfig{
		Bounds:        bounds,
		MaxIterations: 100, // Default value, can be made configurable
		NInitialPoints: 10, // Default value, can be made configurable
	}

	// Create a cancellable context
	ctx, cancel := context.WithCancel(context.Background())

	// Create optimizer
	optimizer, err := bayesian.NewBayesianOptimizer(config)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create optimizer: %v", err)
	}

	// Create optimization state
	state := &OptimizationState{
		ID:          id,
		Status:      "pending",
		StartTime:   time.Now(),
		Optimizer:   optimizer,
		CancelFunc:  cancel,
		LastUpdated: time.Now(),
	}

	// Store the optimization state
	s.optimizationsMu.Lock()
	s.optimizations[id] = state
	s.optimizationsMu.Unlock()

	// Start optimization in a goroutine
	go s.runOptimization(id, objectiveFn, config, ctx, state)

	return map[string]interface{}{
		"optimization_id": id,
		"status":         "pending",
	}, nil
}

// handleOptimizationStatus handles the optimization.status JSON-RPC method.
// It returns the current status and results of an optimization job.
// Expected parameters: {"optimization_id": "opt_123"}
// Returns: Status object with progress, best solution, and history
func (s *Server) handleOptimizationStatus(params []interface{}) (interface{}, error) {
	if len(params) == 0 {
		return nil, fmt.Errorf("missing required parameters")
	}

	// Parse optimization ID
	paramMap, ok := params[0].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid parameter format, expected object")
	}

	optimizationID, ok := paramMap["optimization_id"].(string)
	if !ok || optimizationID == "" {
		return nil, fmt.Errorf("optimization_id is required")
	}

	s.optimizationsMu.RLock()
	defer s.optimizationsMu.RUnlock()

	state, exists := s.optimizations[optimizationID]
	if !exists {
		return nil, fmt.Errorf("optimization not found")
	}

	response := map[string]interface{}{
		"status":      state.Status,
		"progress":    state.Progress,
		"start_time":  state.StartTime.Format(time.RFC3339),
		"last_update": state.LastUpdated.Format(time.RFC3339),
	}

	// Add end time if available
	if state.EndTime != nil {
		response["end_time"] = state.EndTime.Format(time.RFC3339)
	}

	// Add best solution if available
	if state.BestSolution != nil {
		response["best_solution"] = map[string]interface{}{
			"parameters": state.BestSolution.Parameters,
			"value":      state.BestSolution.Value,
		}
	}

	// Get optimization history if available
	if state.Optimizer != nil {
		history := state.Optimizer.GetHistory()
		if len(history) > 0 {
			historyData := make([]map[string]interface{}, len(history))
			for i, eval := range history {
				historyData[i] = map[string]interface{}{
					"iteration":  eval.Iteration,
					"parameters": eval.Solution.Parameters,
					"value":      eval.Solution.Value,
				}
			}
			response["history"] = historyData
		}

		// Add current best solution from optimizer if available
		if bestSolution := state.Optimizer.GetBestSolution(); bestSolution != nil {
			response["current_best"] = map[string]interface{}{
				"parameters": bestSolution.Parameters,
				"value":      bestSolution.Value,
			}
		}
	}

	return response, nil
}

// handleOptimizationCancel handles the optimization.cancel JSON-RPC method.
// It cancels a running optimization job.
// Expected parameters: {"optimization_id": "opt_123"}
// Returns: nil on success, error on failure
func (s *Server) handleOptimizationCancel(params []interface{}) error {
	if len(params) == 0 {
		return fmt.Errorf("missing required parameters")
	}

	// Parse optimization ID
	paramMap, ok := params[0].(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid parameter format, expected object")
	}

	optimizationID, ok := paramMap["optimization_id"].(string)
	if !ok || optimizationID == "" {
		return fmt.Errorf("optimization_id is required")
	}

	s.optimizationsMu.Lock()
	defer s.optimizationsMu.Unlock()

	state, exists := s.optimizations[optimizationID]
	if !exists {
		return fmt.Errorf("optimization not found")
	}

	switch state.Status {
	case "completed", "failed", "cancelled":
		// Already in a terminal state
		return fmt.Errorf("cannot cancel optimization with status: %s", state.Status)
	}

	// Cancel the optimization
	if state.CancelFunc != nil {
		state.CancelFunc()
	}

	// Update state
	state.Status = "cancelled"
	now := time.Now()
	state.EndTime = &now
	state.LastUpdated = now

	// Log the cancellation
	s.logger.Info("Optimization cancelled", map[string]interface{}{
		"optimization_id": optimizationID,
	})

	return nil
}

// respondWithError sends a JSON-RPC 2.0 error response
func (s *Server) respondWithError(w http.ResponseWriter, code int, message string, id interface{}) {
	s.logger.Error("Request error", map[string]interface{}{
		"status":  code,
		"message": message,
	})

	response := map[string]interface{}{
		"jsonrpc": "2.0",
		"error": map[string]interface{}{
			"code":    code,
			"message": message,
		},
		"id": id,
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(response)
}

// runOptimization executes the optimization process in a goroutine
func (s *Server) runOptimization(id, objectiveFn string, config optimization.OptimizerConfig, ctx context.Context, state *OptimizationState) {
	// Update state to running
	s.optimizationsMu.Lock()
	state.Status = "running"
	s.optimizations[id] = state
	s.optimizationsMu.Unlock()

	// Create objective function
	// In a real implementation, this would parse and evaluate the objective function
	// For now, we'll use a simple quadratic function as an example
	objective := func(x []float64) (float64, error) {
		// Simple quadratic function: f(x) = x[0]^2 + x[1]^2 + ...
		sum := 0.0
		for _, v := range x {
			sum += v * v
		}
		return sum, nil
	}

	// Set the objective function in the config
	config.Objective = objective

	// Run optimization
	result, err := state.Optimizer.Optimize(ctx, config)

	// Update state with results
	s.optimizationsMu.Lock()
	defer s.optimizationsMu.Unlock()

	if err != nil {
		s.logger.Error("Optimization failed", map[string]interface{}{
			"optimization_id": id,
			"error":           err.Error(),
		})
		state.Status = "failed"
	} else {
		state.Status = "completed"
		state.BestSolution = result.BestSolution
	}

	now := time.Now()
	state.EndTime = &now
	state.LastUpdated = now
}

// Close cleans up resources
func (s *Server) Close() error {
	// Cancel all running optimizations
	s.optimizationsMu.Lock()
	defer s.optimizationsMu.Unlock()

	for _, opt := range s.optimizations {
		if opt.CancelFunc != nil {
			opt.CancelFunc()
		}
	}
	return nil
}

// handleOptimize handles the HTTP POST /optimize endpoint for starting a new optimization
func (s *Server) handleOptimize(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse request body
	var reqBody struct {
		Objective string      `json:"objective"`
		Bounds    [][]float64 `json:"bounds"`
	}

	if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return
	}

	// Call the JSON-RPC handler
	result, err := s.handleOptimizeStart([]interface{}{map[string]interface{}{
		"objective": reqBody.Objective,
		"bounds":    reqBody.Bounds,
	}})

	// Handle response
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(result)
}

// handleStatus handles the HTTP GET /status/:id endpoint for checking optimization status
func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract optimization ID from URL
	optimizationID := chi.URLParam(r, "id")
	if optimizationID == "" {
		http.Error(w, "Missing optimization ID", http.StatusBadRequest)
		return
	}

	// Call the JSON-RPC handler
	result, err := s.handleOptimizationStatus([]interface{}{map[string]interface{}{
		"optimization_id": optimizationID,
	}})

	// Handle response
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(result)
}

// handleCancel handles the HTTP POST /cancel/:id endpoint for canceling an optimization
func (s *Server) handleCancel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract optimization ID from URL
	optimizationID := chi.URLParam(r, "id")
	if optimizationID == "" {
		http.Error(w, "Missing optimization ID", http.StatusBadRequest)
		return
	}

	// Call the JSON-RPC handler
	err := s.handleOptimizationCancel([]interface{}{map[string]interface{}{
		"optimization_id": optimizationID,
	}})

	// Handle response
	w.Header().Set("Content-Type", "application/json")
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{
		"status": "cancellation requested",
	})
}
