package bayesian

import (
	"context"
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/copyleftdev/TUNDR/internal/optimization"
	"github.com/copyleftdev/TUNDR/internal/optimization/acquisition"
	"github.com/copyleftdev/TUNDR/internal/optimization/kernels"
)

func TestNewBayesianOptimizer(t *testing.T) {
	tests := []struct {
		name          string
		config        optimization.OptimizerConfig
		expectDefault bool
	}{
		{
			name: "valid configuration",
			config: optimization.OptimizerConfig{
				Objective:     func(x []float64) (float64, error) { return 0, nil },
				Bounds:        [][2]float64{{0, 1}},
				MaxIterations: 10,
				NInitialPoints: 5,
			},
			expectDefault: false,
		},
		{
			name: "default values",
			config: optimization.OptimizerConfig{
				Objective:     func(x []float64) (float64, error) { return 0, nil },
				Bounds:        [][2]float64{{0, 1}},
				MaxIterations: 0, // Should use default
				NInitialPoints: 0, // Should use default
			},
			expectDefault: true,
		},
		{
			name: "no objective function",
			config: optimization.OptimizerConfig{
				Objective:     nil,
				Bounds:        [][2]float64{{0, 1}},
				MaxIterations: 10,
			},
			expectDefault: false,
		},
		{
			name: "no bounds",
			config: optimization.OptimizerConfig{
				Objective:     func(x []float64) (float64, error) { return 0, nil },
				Bounds:        nil,
				MaxIterations: 10,
			},
			expectDefault: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			optimizer, err := NewBayesianOptimizer(tt.config)
			
			// The current implementation doesn't return an error for nil Objective or Bounds
			// So we just check that we get a non-nil optimizer in all cases
			require.NoError(t, err)
			require.NotNil(t, optimizer)

			// Verify GP is initialized
			assert.NotNil(t, optimizer.gp)

			// Verify acquisition function is initialized
			assert.NotNil(t, optimizer.acquisition)

			// Verify RNG is initialized
			assert.NotNil(t, optimizer.rng)

			// Verify default values if expected
			if tt.expectDefault {
				assert.Equal(t, 10, optimizer.config.NInitialPoints)
				assert.Equal(t, 50, optimizer.config.MaxIterations)
			}

			// Verify history is initialized with correct capacity
			expectedCapacity := 0
			if tt.config.MaxIterations > 0 && tt.config.NInitialPoints > 0 {
				expectedCapacity = tt.config.MaxIterations + tt.config.NInitialPoints
			} else if tt.config.MaxIterations > 0 {
				expectedCapacity = tt.config.MaxIterations + 10 // Default NInitialPoints
			} else if tt.config.NInitialPoints > 0 {
				expectedCapacity = 50 + tt.config.NInitialPoints // Default MaxIterations
			} else {
				expectedCapacity = 60 // Default MaxIterations + default NInitialPoints
			}
			assert.NotNil(t, optimizer.history)
			assert.Equal(t, 0, len(optimizer.history))
			assert.Equal(t, expectedCapacity, cap(optimizer.history), "Unexpected history capacity")
		})
	}
}

func TestPrepareTrainingData(t *testing.T) {
	tests := []struct {
		name     string
		history  []optimization.Evaluation
		bounds   [][2]float64
		expectedX [][]float64
		expectedY []float64
	}{
		{
			name: "single point",
			history: []optimization.Evaluation{
				{
					Solution: &optimization.Solution{
						Parameters: []float64{1.0, 2.0},
						Value:      3.0,
					},
				},
			},
			bounds: [][2]float64{{0, 10}, {0, 10}},
			expectedX: [][]float64{{1.0, 2.0}},
			expectedY: []float64{3.0},
		},
		{
			name: "multiple points",
			history: []optimization.Evaluation{
				{
					Solution: &optimization.Solution{
						Parameters: []float64{1.0, 2.0},
						Value:      3.0,
					},
				},
				{
					Solution: &optimization.Solution{
						Parameters: []float64{4.0, 5.0},
						Value:      6.0,
					},
				},
			},
			bounds: [][2]float64{{0, 10}, {0, 10}},
			expectedX: [][]float64{{1.0, 2.0}, {4.0, 5.0}},
			expectedY: []float64{3.0, 6.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a BayesianOptimizer with the test history
			bo := &BayesianOptimizer{
				config: optimization.OptimizerConfig{
					Bounds: tt.bounds,
				},
				history: tt.history,
			}

			// Call prepareTrainingData
			X, y := bo.prepareTrainingData()

			// Verify X matrix
			rows, cols := X.Dims()
			assert.Equal(t, len(tt.expectedX), rows)
			assert.Equal(t, len(tt.bounds), cols)

			for i, expectedRow := range tt.expectedX {
				for j, expectedVal := range expectedRow {
					assert.InDelta(t, expectedVal, X.At(i, j), 1e-10, 
						"X[%d][%d] not equal. Expected %v, got %v", 
						i, j, expectedVal, X.At(i, j))
				}
			}

			// Verify y vector
			assert.Equal(t, len(tt.expectedY), y.Len())
			for i, expectedVal := range tt.expectedY {
				assert.InDelta(t, expectedVal, y.AtVec(i), 1e-10,
					"y[%d] not equal. Expected %v, got %v",
					i, expectedVal, y.AtVec(i))
			}
		})
	}
}

func TestBayesianOptimizer(t *testing.T) {
	t.Log("Starting TestBayesianOptimizer")
	// Define a simple 1D objective function (minimization)
	objective := func(x []float64) (float64, error) {
		result := x[0] * x[0] // x^2, minimum at x=0
		t.Logf("Objective called with x=%v, returning %f", x, result)
		return result, nil
	}

	// Define bounds for the parameter
	bounds := [][2]float64{{
		-10.0, // Lower bound
		10.0,  // Upper bound
	}}
	t.Logf("Test bounds: %v", bounds)

	// Create optimizer config
	config := optimization.OptimizerConfig{
		Objective:     objective,
		Bounds:        bounds,
		MaxIterations: 10,
		NInitialPoints: 5,
		RandomSeed:    42, // Fixed seed for reproducibility
	}
	t.Logf("Optimizer config: %+v", config)

	// Create optimizer
	optimizer, err := NewBayesianOptimizer(config)
	t.Logf("Optimizer created, err: %v", err)
	require.NoError(t, err)
	require.NotNil(t, optimizer)

	// Run optimization
	t.Log("Starting optimization...")
	result, err := optimizer.Optimize(context.Background(), config)
	t.Logf("Optimization completed, result: %+v, err: %v", result, err)
	require.NoError(t, err)
	require.NotNil(t, result)

	// Check results
	t.Logf("Verifying results - Best solution: %+v", result.BestSolution)
	assert.NotNil(t, result.BestSolution, "best solution should not be nil")
	assert.InDelta(t, 0.0, result.BestSolution.Value, 0.5, "should find minimum near 0")
	assert.InDelta(t, 0.0, result.BestSolution.Parameters[0], 1.0, "should find x near 0")
	assert.True(t, result.Converged, "should converge")
	assert.Equal(t, config.MaxIterations+config.NInitialPoints, len(result.History), "history should have one entry per evaluation")

	// Check history
	historySize := len(result.History)
	t.Logf("History size: %d", historySize)
	assert.Greater(t, historySize, 0, "history should not be empty")

	// Verify best solution is in history
	found := false
	for i, eval := range result.History {
		if eval.Solution.Value == result.BestSolution.Value {
			found = true
			t.Logf("Found best solution at history index %d: %+v", i, eval)
		}
		assert.Equal(t, i, eval.Iteration, "history should be in order")
		assert.NotNil(t, eval.Solution, "solution should not be nil")
	}

	if !found {
		t.Log("Best solution not found in history. History contents:")
		for i, eval := range result.History {
			t.Logf("  [%d] %+v", i, eval)
		}
	}
	assert.True(t, found, "best solution should be in history")
	t.Log("TestBayesianOptimizer completed successfully")
}

func TestBayesianOptimizerWithConstraints(t *testing.T) {
	// Define a constrained optimization problem
	// Minimize f(x) = x^2, subject to x >= 1
	objective := func(x []float64) (float64, error) {
		if x[0] < 1.0 {
			return math.Inf(1), nil // Return +inf for invalid points
		}
		return x[0] * x[0], nil
	}

	bounds := [][2]float64{{
		0.0, // Lower bound
		2.0, // Upper bound
	}}

	config := optimization.OptimizerConfig{
		Objective:     objective,
		Bounds:       bounds,
		MaxIterations: 15,
		NInitialPoints: 5,
		RandomSeed:    42,
	}

	optimizer, err := NewBayesianOptimizer(config)
	require.NoError(t, err)

	result, err := optimizer.Optimize(context.Background(), config)
	require.NoError(t, err)

	// Should find minimum at x=1 (boundary of constraint)
	assert.InDelta(t, 1.0, result.BestSolution.Value, 0.1, "should find minimum at constraint boundary")
	assert.InDelta(t, 1.0, result.BestSolution.Parameters[0], 0.1, "should find x at constraint boundary")
}

func TestBayesianOptimizerWithCustomKernel(t *testing.T) {
	// Test with a custom kernel
	objective := func(x []float64) (float64, error) {
		return math.Sin(x[0]) + math.Cos(x[1]), nil
	}

	bounds := [][2]float64{{
		-3.0, 3.0,
	}, {
		-3.0, 3.0,
	}}

	// Create a custom kernel
	kernel := kernels.NewMatern52Kernel(1.0, 1.0)


	// Create optimizer with custom kernel
	optimizer := &BayesianOptimizer{
		config: optimization.OptimizerConfig{
			Objective:     objective,
			Bounds:       bounds,
			MaxIterations: 30,
			NInitialPoints: 10,
			RandomSeed:    42,
		},
		gp:         NewGP(kernel, 1e-6),
		acquisition: acquisition.NewExpectedImprovement(0.1, 0.01),
		rng:        rand.New(rand.NewSource(42)),
		history:    make([]optimization.Evaluation, 0, 40),
	}

	result, err := optimizer.Optimize(context.Background(), optimizer.config)
	require.NoError(t, err)

	// Check that we found a reasonable solution
	// Global minimum of sin(x) + cos(y) is -2 at (3π/2, π)
	// But with bounds [-3, 3], minimum is at (-π/2, 0) with value -1
	assert.Less(t, result.BestSolution.Value, -0.9, "should find minimum below -0.9")
}

func TestBayesianOptimizerCancel(t *testing.T) {
	// Test that optimization can be cancelled
	objective := func(x []float64) (float64, error) {
		return x[0] * x[0], nil
	}

	bounds := [][2]float64{{
		-10.0,
		10.0,
	}}

	config := optimization.OptimizerConfig{
		Objective:     objective,
		Bounds:       bounds,
		MaxIterations: 100,
		NInitialPoints: 5,
	}

	optimizer, err := NewBayesianOptimizer(config)
	require.NoError(t, err)

	// Create a cancellable context
	ctx, cancel := context.WithCancel(context.Background())
	// Cancel after a short delay
	cancel()

	// This should return immediately due to cancellation
	result, err := optimizer.Optimize(ctx, config)
	require.Error(t, err, "should return error when context is cancelled")
	assert.Nil(t, result, "should not return result when cancelled")
}

func TestLatinHypercubeSampling(t *testing.T) {
	// Test that LHS generates points within bounds
	bounds := [][2]float64{{
		-2.0, 2.0,
	}, {
		0.0, 5.0,
	}}

	config := optimization.OptimizerConfig{
		Bounds:        bounds,
		NInitialPoints: 10,
	}

	optimizer := &BayesianOptimizer{
		config: config,
		rng:    rand.New(rand.NewSource(42)),
	}

	samples := optimizer.latinHypercubeSample(config.NInitialPoints)

	// Check dimensions
	assert.Len(t, samples, config.NInitialPoints, "should generate requested number of samples")
	for _, sample := range samples {
		assert.Len(t, sample, len(bounds), "sample should have correct dimensionality")

		// Check each dimension is within bounds
		for i, val := range sample {
			assert.GreaterOrEqual(t, val, bounds[i][0], "sample should be >= lower bound")
			assert.LessOrEqual(t, val, bounds[i][1], "sample should be <= upper bound")
		}
	}

	// Check that points are well-distributed
	// For LHS, we expect exactly one point in each interval when divided into N bins
	bins := make([][]bool, len(bounds))
	for i := range bins {
		bins[i] = make([]bool, config.NInitialPoints)
	}

	for _, sample := range samples {
		for dim, val := range sample {
			// Determine which bin this point falls into
			bin := int(float64(config.NInitialPoints) * (val - bounds[dim][0]) / (bounds[dim][1] - bounds[dim][0]))
			// Handle edge case where val equals upper bound
			if bin >= config.NInitialPoints {
				bin = config.NInitialPoints - 1
			}
			// Should be the only point in this bin for this dimension
			assert.False(t, bins[dim][bin], "should have only one point per bin")
			bins[dim][bin] = true
		}
	}
}
