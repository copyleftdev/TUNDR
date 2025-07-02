package optimization

import (
	"context"
)

// Optimizer defines the interface for optimization algorithms
type Optimizer interface {
	// Optimize runs the optimization process
	Optimize(ctx context.Context, config OptimizerConfig) (*OptimizationResult, error)

	// GetBestSolution returns the best solution found so far
	GetBestSolution() *Solution

	// GetHistory returns the history of evaluations
	GetHistory() []Evaluation

	// Stop gracefully stops the optimization process
	Stop()
}

// OptimizerConfig contains configuration for the optimizer
type OptimizerConfig struct {
	// Objective function to optimize
	Objective ObjectiveFunction

	// Bounds for each dimension [min, max]
	Bounds [][2]float64

	// Maximum number of iterations
	MaxIterations int

	// Number of initial random points to evaluate
	NInitialPoints int

	// Random seed for reproducibility
	RandomSeed int64

	// Verbose logging
	Verbose bool
}

// ObjectiveFunction defines the function to be optimized
type ObjectiveFunction func([]float64) (float64, error)

// Solution represents a solution in the optimization space
type Solution struct {
	Parameters []float64
	Value      float64
}

// Evaluation represents a single evaluation of the objective function
type Evaluation struct {
	Iteration int
	Solution  *Solution
	Error     error
}

// OptimizationResult contains the result of an optimization run
type OptimizationResult struct {
	BestSolution *Solution
	History     []Evaluation
	Iterations  int
	Converged   bool
}
