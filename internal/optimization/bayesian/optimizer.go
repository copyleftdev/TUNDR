package bayesian

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/copyleftdev/TUNDR/internal/optimization"
	"github.com/copyleftdev/TUNDR/internal/optimization/bayesian/acquisition"
	"github.com/copyleftdev/TUNDR/internal/optimization/bayesian/kernels"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"
)

// BayesianOptimizer implements Bayesian Optimization
// AcquisitionFunction defines the interface for acquisition functions
type AcquisitionFunction interface {
	Acquire(gp *GP, X *mat.Dense, y *mat.VecDense, bounds [][2]float64, rng *rand.Rand) ([]float64, error)
}

type BayesianOptimizer struct {
	// Configuration
	config BayesianOptimizerConfig

	// Gaussian Process model
	gp *GP

	// Acquisition function
	acquisition AcquisitionFunction

	// Random number generator
	rand *rand.Rand

	// History of evaluations
	history []optimization.Evaluation

	// Best solution found so far
	bestSolution *optimization.Solution

	// For cancellation
	cancel context.CancelFunc

	// Best values at each iteration
	bestValues []float64
}

// BayesianOptimizerConfig extends the base optimization config with Bayesian-specific settings
type BayesianOptimizerConfig struct {
	// Base optimization config
	optimization.OptimizerConfig
	
	// Tolerance is the minimum improvement required to consider the optimization converged
	Tolerance float64 `json:"tolerance"`
	// Patience is the number of iterations to wait before declaring convergence if no improvement is made
	Patience int `json:"patience"`
}

// DefaultBayesianOptimizerConfig returns the default configuration for the BayesianOptimizer
func DefaultBayesianOptimizerConfig() *BayesianOptimizerConfig {
	return &BayesianOptimizerConfig{
		OptimizerConfig: optimization.OptimizerConfig{
			MaxIterations:  100,
			NInitialPoints: 10,
			RandomSeed:     42,
		},
		Tolerance: 1e-6, // Default tolerance for convergence
		Patience:  5,    // Number of iterations to wait before declaring convergence
	}
}

// NewBayesianOptimizer creates a new Bayesian Optimizer
func NewBayesianOptimizer(cfg optimization.OptimizerConfig) (*BayesianOptimizer, error) {
	if cfg.MaxIterations < 1 {
		cfg.MaxIterations = 50 // Default value
	}

	if cfg.NInitialPoints < 1 {
		cfg.NInitialPoints = 10 // Default value
	}

	// Initialize random number generator
	rng := rand.New(rand.NewSource(cfg.RandomSeed))
	if cfg.RandomSeed == 0 {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	// Initialize GP with default config
	kernel := kernels.NewMatern52Kernel(1.0, 1.0)
	gp := NewGP(kernel, 1e-6) // Small noise for numerical stability

	return &BayesianOptimizer{
		config: BayesianOptimizerConfig{
			OptimizerConfig: cfg,
			Tolerance:       1e-6, // Default tolerance
			Patience:        5,    // Default patience
		},
		gp:             gp,
		rand:           rng,
		history:        make([]optimization.Evaluation, 0, cfg.MaxIterations+cfg.NInitialPoints),
		acquisition:    acquisition.NewExpectedImprovement(math.Inf(1), 0.01),
		bestSolution:   nil,
		bestValues:     make([]float64, 0),
	}, nil
}

// Optimize runs the Bayesian optimization
func (bo *BayesianOptimizer) Optimize(ctx context.Context, cfg optimization.OptimizerConfig) (*optimization.OptimizationResult, error) {
	// Update config if provided
	if cfg.Objective != nil {
		bo.config.Objective = cfg.Objective
	}
	if cfg.Bounds != nil {
		bo.config.Bounds = cfg.Bounds
	}
	if cfg.MaxIterations > 0 {
		bo.config.MaxIterations = cfg.MaxIterations
	}
	if cfg.NInitialPoints > 0 {
		bo.config.NInitialPoints = cfg.NInitialPoints
	}

	// Validate configuration
	if bo.config.Objective == nil {
		return nil, fmt.Errorf("objective function is required")
	}
	if len(bo.config.Bounds) == 0 {
		return nil, fmt.Errorf("parameter bounds are required")
	}

	// Reset state
	bo.history = bo.history[:0]
	bo.bestSolution = nil
	bo.bestValues = bo.bestValues[:0]

	// Create a cancellable context
	ctx, bo.cancel = context.WithCancel(ctx)
	defer bo.cancel()

	// Initial random sampling (Latin Hypercube Sampling)
	initialPoints := bo.latinHypercubeSample(bo.config.NInitialPoints)

	// Evaluate initial points
	for i, x := range initialPoints {
		select {
		case <-ctx.Done():
			// When context is cancelled, return nil result with the context error
			return nil, ctx.Err()
		default:
			// Continue with optimization
		}

		// Evaluate the objective function
		value, err := bo.config.Objective(x)
		if err != nil {
			return nil, fmt.Errorf("error evaluating objective function: %v", err)
		}

		// Update best solution
		bo.updateBestSolution(x, value)

		// Record evaluation
		eval := optimization.Evaluation{
			Iteration: i,
			Solution: &optimization.Solution{
				Parameters: x,
				Value:      value,
			},
		}
		bo.history = append(bo.history, eval)
	}

	// Main optimization loop
	for i := 0; i < bo.config.MaxIterations; i++ {
		select {
		case <-ctx.Done():
			// When context is cancelled, return nil result with the context error
			return nil, ctx.Err()
		default:
			// Continue with optimization
		}

		// Fit GP to current data
		X, y := bo.prepareTrainingData()
		err := bo.gp.Fit(X, y)
		if err != nil {
			return nil, fmt.Errorf("error fitting GP: %v", err)
		}

		// Select next point using acquisition function
		nextPoint, err := bo.acquisition.Acquire(bo.gp, X, y, bo.config.Bounds, bo.rand)
		if err != nil {
			return nil, fmt.Errorf("failed to select next point: %v", err)
		}

		// Evaluate the objective function at the next point
		value, err := bo.config.Objective(nextPoint)
		if err != nil {
			return nil, fmt.Errorf("error evaluating objective function: %v", err)
		}

		// Update best solution
		bo.updateBestSolution(nextPoint, value)

		// Record evaluation
		eval := optimization.Evaluation{
			Iteration: i + bo.config.NInitialPoints,
			Solution: &optimization.Solution{
				Parameters: nextPoint,
				Value:      value,
			},
		}
		bo.history = append(bo.history, eval)

		// Check for convergence after initial points
		if i >= bo.config.NInitialPoints + bo.config.Patience {
			// Calculate the best value in the last N iterations (patience window)
			bestInWindow := bo.history[len(bo.history)-1].Solution.Value
			for j := 2; j <= bo.config.Patience; j++ {
				if len(bo.history)-j >= 0 && bo.history[len(bo.history)-j].Solution.Value < bestInWindow {
					bestInWindow = bo.history[len(bo.history)-j].Solution.Value
				}
			}

			// Calculate the best value before the patience window
			bestBefore := bo.bestSolution.Value
			if len(bo.history) > bo.config.Patience {
				bestBefore = bo.history[len(bo.history)-bo.config.Patience-1].Solution.Value
			}

			// Check if the improvement is below tolerance
			improvement := math.Abs(bestBefore - bestInWindow)
			if improvement < bo.config.Tolerance {
				return &optimization.OptimizationResult{
					BestSolution: bo.bestSolution,
					History:     bo.history,
					Iterations:  i + 1,
					Converged:   true,
				}, nil
			}
		}
	}

	return &optimization.OptimizationResult{
		BestSolution: bo.bestSolution,
		History:     bo.history,
		Iterations:  bo.config.MaxIterations + bo.config.NInitialPoints,
		Converged:   true,
	}, nil
}

// GetBestSolution returns the best solution found so far
func (bo *BayesianOptimizer) GetBestSolution() *optimization.Solution {
	return bo.bestSolution
}

// GetHistory returns the history of evaluations
func (bo *BayesianOptimizer) GetHistory() []optimization.Evaluation {
	return bo.history
}

// Stop stops the optimization process
func (bo *BayesianOptimizer) Stop() {
	if bo.cancel != nil {
		bo.cancel()
	}
}

// updateBestSolution updates the best solution if the new solution is better
func (bo *BayesianOptimizer) updateBestSolution(params []float64, value float64) {
	if bo.bestSolution == nil || value < bo.bestSolution.Value {
		bo.bestSolution = &optimization.Solution{
			Parameters: params,
			Value:      value,
		}
	}
	// Track the best value at each iteration
	bo.bestValues = append(bo.bestValues, bo.bestSolution.Value)
}

// prepareTrainingData prepares the training data for the GP
func (bo *BayesianOptimizer) prepareTrainingData() (*mat.Dense, *mat.VecDense) {
	nSamples := len(bo.history)
	if nSamples == 0 {
		return nil, nil
	}

	nDims := len(bo.history[0].Solution.Parameters)
	X := mat.NewDense(nSamples, nDims, nil)
	y := mat.NewVecDense(nSamples, nil)

	for i, eval := range bo.history {
		for j, val := range eval.Solution.Parameters {
			X.Set(i, j, val)
		}
		y.SetVec(i, eval.Solution.Value)
	}

	return X, y
}

// latinHypercubeSample generates points using Latin Hypercube Sampling
func (bo *BayesianOptimizer) latinHypercubeSample(n int) [][]float64 {
	nDims := len(bo.config.Bounds)
	samples := make([][]float64, n)

	// Generate samples in [0,1]^nDims
	for i := 0; i < nDims; i++ {
		// Generate stratified random samples
		samples1D := make([]float64, n)
		for j := 0; j < n; j++ {
			samples1D[j] = float64(j) + bo.rng.Float64()
		}

		// Shuffle
		bo.rng.Shuffle(n, func(k, l int) {
			samples1D[k], samples1D[l] = samples1D[l], samples1D[k]
		})


		// Scale to [0,1]
		for j := 0; j < n; j++ {
			samples1D[j] = samples1D[j] / float64(n)
		}

		// Scale to parameter bounds and store
		for j := 0; j < n; j++ {
			if i == 0 {
				samples[j] = make([]float64, nDims)
			}
			min, max := bo.config.Bounds[i][0], bo.config.Bounds[i][1]
			samples[j][i] = min + samples1D[j]*(max-min)
		}
	}

	return samples
}

// maximizeAcquisition finds the point that maximizes the acquisition function
func (bo *BayesianOptimizer) maximizeAcquisition() ([]float64, error) {
	nDims := len(bo.config.Bounds)

	// Define the objective function for optimization (minimization)
	objective := func(x []float64) float64 {
		// Predict mean and variance at x
		X := mat.NewDense(1, nDims, x)
		mu, sigmaSq, err := bo.gp.Predict(X)
		if err != nil {
			return math.Inf(1)
		}
		sigma := math.Sqrt(sigmaSq.AtVec(0))

		// Negate because we're minimizing
		return -bo.acquisition.Compute(mu.AtVec(0), sigma)
	}

	// Initial points for optimization (current best + random points)
	nStarts := 5 + int(5*math.Sqrt(float64(nDims)))
	starts := make([][]float64, nStarts)

	// Start from current best
	if bo.bestSolution != nil {
		starts[0] = append([]float64(nil), bo.bestSolution.Parameters...)
	}

	// Add random starts
	for i := 0; i < nStarts; i++ {
		if starts[i] == nil {
			starts[i] = make([]float64, nDims)
			for j := 0; j < nDims; j++ {
				min, max := bo.config.Bounds[j][0], bo.config.Bounds[j][1]
				starts[i][j] = min + bo.rng.Float64()*(max-min)
			}
		}
	}

	// Find best point among all starts
	bestX := make([]float64, nDims)
	bestVal := math.Inf(1)

	// Create a problem with the objective function
	problem := optimize.Problem{
		Func: func(x []float64) float64 {
			// Ensure x is within bounds
			for i := range x {
				x[i] = math.Max(bo.config.Bounds[i][0], math.Min(x[i], bo.config.Bounds[i][1]))
			}
			return objective(x)
		},
	}

	// Create settings for the optimizer
	settings := &optimize.Settings{
		Converger: &optimize.FunctionConverge{
			Absolute:   1e-6,
			Relative:   1e-6,
			Iterations: 100,
		},
	}

	// Try each starting point
	for _, start := range starts {
		// Use Nelder-Mead method which is derivative-free
		method := &optimize.NelderMead{
			Reflection:  1.0,  // Standard reflection coefficient
			Expansion:   2.0,  // Standard expansion coefficient
			Contraction: 0.5,  // Standard contraction coefficient
			Shrink:      0.5,  // Standard shrink coefficient
			SimplexSize: 0.2,  // Size of auto-constructed initial simplex
		}
		
		// Run optimization
		result, err := optimize.Minimize(problem, start, settings, method)

		if err == nil && result.F < bestVal {
			bestVal = result.F
			copy(bestX, result.X)
		}
	}

	return bestX, nil
}
