package bayesian

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize"

	"github.com/copyleftdev/TUNDR/internal/optimization"
	"github.com/copyleftdev/TUNDR/internal/optimization/acquisition"
	"github.com/copyleftdev/TUNDR/internal/optimization/kernels"
)

// BayesianOptimizer implements Bayesian Optimization
type BayesianOptimizer struct {
	// Configuration
	config optimization.OptimizerConfig

	// Gaussian Process model
	gp *GP

	// Acquisition function
	acquisition *acquisition.ExpectedImprovement

	// Random number generator
	rng *rand.Rand

	// Best solution found
	bestSolution *optimization.Solution

	// History of evaluations
	history []optimization.Evaluation

	// For cancellation
	cancel context.CancelFunc
}

// NewBayesianOptimizer creates a new Bayesian Optimizer
func NewBayesianOptimizer(config optimization.OptimizerConfig) (*BayesianOptimizer, error) {
	if config.NInitialPoints < 1 {
		config.NInitialPoints = 10 // Default value
	}

	if config.MaxIterations < 1 {
		config.MaxIterations = 50 // Default value
	}

	// Initialize random number generator
	rng := rand.New(rand.NewSource(config.RandomSeed))
	if config.RandomSeed == 0 {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	// Default kernel: Matern 5/2 with automatic relevance determination
	kernel := kernels.NewMatern52Kernel(1.0, 1.0)

	return &BayesianOptimizer{
		config:     config,
		gp:         NewGP(kernel, 1e-6), // Small noise for numerical stability
		acquisition: acquisition.NewExpectedImprovement(math.Inf(1), 0.01),
		rng:        rng,
		history:    make([]optimization.Evaluation, 0, config.MaxIterations+config.NInitialPoints),
	}, nil
}

// Optimize runs the Bayesian Optimization process
func (bo *BayesianOptimizer) Optimize(ctx context.Context, config optimization.OptimizerConfig) (*optimization.OptimizationResult, error) {
	// Update config if provided
	if config.Objective != nil {
		bo.config = config
	}

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

		// Update acquisition function with best observed value
		bo.acquisition.UpdateBest(bo.bestSolution.Value)

		// Find next point to evaluate by maximizing acquisition function
		nextPoint, err := bo.maximizeAcquisition()
		if err != nil {
			return nil, fmt.Errorf("error maximizing acquisition function: %v", err)
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

		// TODO: Add convergence check
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
			Parameters: append([]float64(nil), params...),
			Value:      value,
		}
	}
}

// prepareTrainingData prepares the training data for the GP
func (bo *BayesianOptimizer) prepareTrainingData() (*mat.Dense, *mat.VecDense) {
	nSamples := len(bo.history)
	nDims := len(bo.config.Bounds)

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
