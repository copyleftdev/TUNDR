package bayesian

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"gonum.org/v1/gonum/mat"
	"github.com/copyleftdev/TUNDR/internal/optimization"
	"github.com/copyleftdev/TUNDR/internal/optimization/acquisition"
	"github.com/copyleftdev/TUNDR/internal/optimization/kernels"
)

// BenchmarkGPFit measures the performance of fitting a Gaussian Process model
func BenchmarkGPFit(b *testing.B) {
	// Set up test data
	nSamples := 100
	nFeatures := 5
	X := mat.NewDense(nSamples, nFeatures, nil)
	y := mat.NewVecDense(nSamples, nil)

	// Fill with random data
	rand.Seed(42) // For reproducibility
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, rand.NormFloat64())
		}
		y.SetVec(i, rand.NormFloat64())
	}

	// Create a new GP
	kernel := kernels.NewMatern52Kernel(1.0, 1.0)
	gp := NewGP(kernel, 1e-6)

	// Run the benchmark
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = gp.Fit(X, y)
	}
}

// BenchmarkGPSample measures the performance of sampling from a fitted GP
func BenchmarkGPSample(b *testing.B) {
	// Set up test data
	nSamples := 100
	nFeatures := 5
	nTest := 50

	// Create training data
	X := mat.NewDense(nSamples, nFeatures, nil)
	y := mat.NewVecDense(nSamples, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, 0.0)
		}
		y.SetVec(i, 0.0)
	}

	// Create test points
	XTest := mat.NewDense(nTest, nFeatures, nil)
	for i := 0; i < nTest; i++ {
		for j := 0; j < nFeatures; j++ {
			XTest.Set(i, j, rand.NormFloat64()*2-1)
		}
	}

	// Create and fit GP
	kernel := kernels.NewMatern52Kernel(1.0, 1.0)
	gp := NewGP(kernel, 1e-6)
	if err := gp.Fit(X, y); err != nil {
		b.Fatalf("Failed to fit GP: %v", err)
	}

	// Run the benchmark
	nSamplesPerRun := 10
	rng := rand.New(rand.NewSource(42))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = gp.Sample(XTest, nSamplesPerRun, rng)
	}
}

// BenchmarkBayesianOptimization measures the performance of a complete Bayesian Optimization run
func BenchmarkBayesianOptimization(b *testing.B) {
	// Define the objective function (Rosenbrock function)
	objective := func(x []float64) (float64, error) {
		return math.Pow(1-x[0], 2) + 100*math.Pow(x[1]-x[0]*x[0], 2), nil
	}

	// Create optimizer configuration
	config := optimization.OptimizerConfig{
		Objective:      objective,
		NInitialPoints: 10,
		MaxIterations:  20,
		Verbose:        false,
	}

	// Run the benchmark
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Create a new optimizer for each iteration
		optimizer, err := NewBayesianOptimizer(config)
		if err != nil {
			b.Fatalf("Failed to create optimizer: %v", err)
		}

		// Run optimization
		_, _ = optimizer.Optimize(nil, config)
	}
}

// BenchmarkKernelMatrixComputation measures the performance of kernel matrix computations
func BenchmarkKernelMatrixComputation(b *testing.B) {
	nSamples := 100
	nFeatures := 5
	X := mat.NewDense(nSamples, nFeatures, nil)

	// Fill with random data
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, 0.0)
		}
	}

	// Create a kernel
	kernel := kernels.NewMatern52Kernel(1.0, 1.0)

	// Run the benchmark
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Compute kernel matrix manually since ComputeMatrix doesn't exist
		n, _ := X.Dims()
		K := mat.NewSymDense(n, nil)
		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				xi := X.RawRowView(i)
				xj := X.RawRowView(j)
				K.SetSym(i, j, kernel.Eval(xi, xj))
			}
		}
	}
}

// BenchmarkAcquisitionFunction measures the performance of acquisition function evaluations
func BenchmarkAcquisitionFunction(b *testing.B) {
	// Set up a test logger
	config := zap.NewDevelopmentConfig()
	config.EncoderConfig.EncodeLevel = zapcore.CapitalColorLevelEncoder
	logger, err := config.Build()
	require.NoError(b, err)
	defer logger.Sync()

	// Initialize test parameters
	nSamples := 100
	nFeatures := 5
	X := mat.NewDense(nSamples, nFeatures, nil)

	// Fill with random data
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, 0.0)
		}
	}

	// Create a GP and fit it
	kernel := kernels.NewMatern52Kernel(1.0, 1.0)
	gp := NewGP(kernel, 1e-6)
	
	// Set up test data
	y := mat.NewVecDense(nSamples, nil)
	for i := 0; i < nSamples; i++ {
		y.SetVec(i, rand.NormFloat64())
	}

	// Fit the model
	err = gp.Fit(X, y)
	require.NoError(b, err, "Failed to fit GP model")

	// Context is not used in the current benchmark

	// For the benchmark, we'll use a reasonable best value
	bestValue := -1.0
	xi := 0.01
	acq := acquisition.NewExpectedImprovement(bestValue, xi)

	// Test points
	nTest := 100
	XTest := mat.NewDense(nTest, nFeatures, nil)
	for i := 0; i < nTest; i++ {
		for j := 0; j < nFeatures; j++ {
			XTest.Set(i, j, rand.NormFloat64()*2-1)
		}
	}

	// Pre-allocate memory for predictions
	muVec := mat.NewVecDense(nTest, nil)
	sigmaVec := mat.NewVecDense(nTest, nil)

	// Run the benchmark
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Reset vectors
		muVec.Zero()
		sigmaVec.Zero()

		// Make predictions for all test points
		for j := 0; j < nTest; j++ {
			// Get prediction for the test point
			xMat := mat.NewDense(1, nFeatures, XTest.RawRowView(j))
			mu, sigmaSq, err := gp.Predict(xMat)
			if err != nil {
				b.Fatalf("Prediction failed: %v", err)
			}

			// Store results
			muVec.SetVec(j, mu.AtVec(0))
			sigmaVec.SetVec(j, math.Sqrt(math.Max(sigmaSq.AtVec(0), 0)))
		}

		// Compute acquisition function values
		for j := 0; j < nTest; j++ {
			_ = acq.Compute(muVec.AtVec(j), sigmaVec.AtVec(j))
		}
	}
}
