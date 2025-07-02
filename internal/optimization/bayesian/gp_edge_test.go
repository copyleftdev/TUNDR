package bayesian

import (
	"math"
	"math/rand"
	"testing"
	"time"


	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tundr/mcp-optimization-server/internal/optimization/kernels"
	"gonum.org/v1/gonum/mat"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

func TestGPEdgeCases(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	t.Run("high_dimensional_input", func(t *testing.T) {
		// Test with high-dimensional input
		nSamples := 10
		nFeatures := 100 // High-dimensional input
		
		// Generate random high-dimensional data
		X := mat.NewDense(nSamples, nFeatures, nil)
		y := mat.NewVecDense(nSamples, nil)
		for i := 0; i < nSamples; i++ {
			for j := 0; j < nFeatures; j++ {
				X.Set(i, j, rand.NormFloat64())
			}
			y.SetVec(i, rand.NormFloat64())
		}

		// Create and fit GP
		kernel := kernels.NewRBFKernel(1.0, 1.0)
		gp := NewGP(kernel, 1e-6)
		
		err := gp.Fit(X, y)
		require.NoError(t, err, "should not error with high-dimensional input")
		
		// Test prediction
		XTest := mat.NewDense(1, nFeatures, nil)
		for j := 0; j < nFeatures; j++ {
			XTest.Set(0, j, rand.NormFloat64())
		}
		
		mean, std, err := gp.Predict(XTest)
		require.NoError(t, err, "prediction should succeed")
		assert.Equal(t, 1, mean.Len(), "mean should have length 1")
		assert.Equal(t, 1, std.Len(), "std should have length 1")
	})

	t.Run("large_number_of_samples", func(t *testing.T) {
		// Test with a large number of samples
		nSamples := 1000
		nFeatures := 5
		
		// Generate random data
		X := mat.NewDense(nSamples, nFeatures, nil)
		y := mat.NewVecDense(nSamples, nil)
		for i := 0; i < nSamples; i++ {
			for j := 0; j < nFeatures; j++ {
				X.Set(i, j, rand.NormFloat64())
			}
			y.SetVec(i, rand.NormFloat64())
		}

		// Create and fit GP
		kernel := kernels.NewRBFKernel(1.0, 1.0)
		gp := NewGP(kernel, 1e-6)
		
		err := gp.Fit(X, y)
		require.NoError(t, err, "should handle large number of samples")
	})

	t.Run("extreme_kernel_parameters", func(t *testing.T) {
		tests := []struct {
			name          string
			lengthScale   float64
			variance      float64
			shouldSucceed bool
		}{
			{"very_small_length_scale", 1e-10, 1.0, true},
			{"very_large_length_scale", 1e10, 1.0, true},
			{"very_small_variance", 1.0, 1e-10, true},
			{"very_large_variance", 1.0, 1e10, true},
			{"zero_length_scale", 0.0, 1.0, false},
			{"negative_length_scale", -1.0, 1.0, false},
			{"zero_variance", 1.0, 0.0, false},
			{"negative_variance", 1.0, -1.0, false},
		}

		// Simple test data
		nSamples := 5
		nFeatures := 2
		X := mat.NewDense(nSamples, nFeatures, []float64{
			0, 0,
			1, 1,
			2, 2,
			3, 3,
			4, 4,
		})
		y := mat.NewVecDense(nSamples, []float64{0, 1, 4, 9, 16})

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if tt.shouldSucceed {
					kernel := kernels.NewRBFKernel(tt.lengthScale, tt.variance)
					gp := NewGP(kernel, 1e-6)
					
					err := gp.Fit(X, y)
					assert.NoError(t, err, "should handle kernel parameters: %v", tt.name)
					
					// Test prediction
					XTest := mat.NewDense(1, nFeatures, []float64{1.5, 1.5})
					mean, std, err := gp.Predict(XTest)
					assert.NoError(t, err, "prediction should succeed with %v", tt.name)
					assert.Equal(t, 1, mean.Len(), "mean should have length 1")
					assert.Equal(t, 1, std.Len(), "std should have length 1")
				} else {
					// For invalid parameters, expect a panic from NewRBFKernel
					assert.Panics(t, func() {
						kernels.NewRBFKernel(tt.lengthScale, tt.variance)
					}, "should panic with invalid parameters: %v", tt.name)
				}
			})
		}
	})

	t.Run("different_kernel_types", func(t *testing.T) {
		nSamples := 10
		nFeatures := 2
		
		// Generate random data
		X := mat.NewDense(nSamples, nFeatures, nil)
		y := mat.NewVecDense(nSamples, nil)
		for i := 0; i < nSamples; i++ {
			for j := 0; j < nFeatures; j++ {
				X.Set(i, j, rand.NormFloat64())
			}
			y.SetVec(i, rand.NormFloat64())
		}

		tests := []struct {
			name    string
			kernel  kernels.Kernel
		}{
			{"rbf", kernels.NewRBFKernel(1.0, 1.0)},
			{"matern52", kernels.NewMatern52Kernel(1.0, 1.0)},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				gp := NewGP(tt.kernel, 1e-6)
				
				err := gp.Fit(X, y)
				require.NoError(t, err, "%s kernel should fit successfully", tt.name)
				
				// Test prediction
				XTest := mat.NewDense(1, nFeatures, []float64{0.5, 0.5})
				mean, std, err := gp.Predict(XTest)
				
				assert.NoError(t, err, "%s kernel prediction should succeed", tt.name)
				assert.Equal(t, 1, mean.Len(), "%s kernel: mean should have length 1", tt.name)
				assert.Equal(t, 1, std.Len(), "%s kernel: std should have length 1", tt.name)
				assert.Greater(t, math.Abs(mean.AtVec(0)), 0.0, "%s kernel: mean should be non-zero", tt.name)
				assert.GreaterOrEqual(t, std.AtVec(0), 0.0, "%s kernel: std should be non-negative", tt.name)
			})
		}
	})
}

func TestGPSamplingEdgeCases(t *testing.T) {
	rand.Seed(time.Now().UnixNano())
	t.Run("sample_from_prior", func(t *testing.T) {
		// Test sampling from the GP prior
		nSamples := 5
		nFeatures := 2
		
		// Generate some test data
		X := mat.NewDense(nSamples, nFeatures, nil)
		y := mat.NewVecDense(nSamples, nil)
		for i := 0; i < nSamples; i++ {
			for j := 0; j < nFeatures; j++ {
				X.Set(i, j, rand.NormFloat64())
			}
			y.SetVec(i, rand.NormFloat64())
		}

		// Create and fit GP
		kernel := kernels.NewRBFKernel(1.0, 1.0)
		gp := NewGP(kernel, 1e-6)
		
		// Sample from the prior
		// First fit the model with some data
		err := gp.Fit(X, y)
		require.NoError(t, err, "should fit the model")
		
		rng := rand.New(rand.NewSource(42))
		samples, err := gp.Sample(X, 3, rng) // Draw 3 samples
		require.NoError(t, err, "should sample from prior")
		// Samples should have shape (nPoints, nSamples)
		nPoints, nSamples := samples.Dims()
		nInputPoints := X.RawMatrix().Rows
		// The number of points should match the number of input points
		assert.Equal(t, nInputPoints, nPoints, "number of points should match input rows")
		// The number of samples should match the number of samples requested (3)
		assert.Equal(t, 3, nSamples, "should return 3 samples")
		
		// Check that samples are not all zeros
		hasNonZero := false
		for i := 0; i < nPoints; i++ {
			for j := 0; j < nSamples; j++ {
				if samples.At(i, j) != 0 {
					hasNonZero = true
					break
				}
			}
			if hasNonZero {
				break
			}
		}
		assert.True(t, hasNonZero, "samples should have non-zero values")
	})

	t.Run("sample_from_posterior", func(t *testing.T) {
		// Test sampling from the posterior after fitting
		nTrain := 5
		nTest := 3
		nFeatures := 2
		
		// Training data
		X := mat.NewDense(nTrain, nFeatures, []float64{
			0, 0,
			1, 1,
			2, 2,
			3, 3,
			4, 4,
		})
		y := mat.NewVecDense(nTrain, []float64{0, 1, 4, 9, 16})

		// Test points
		XTest := mat.NewDense(nTest, nFeatures, []float64{
			0.5, 0.5,
			1.5, 1.5,
			2.5, 2.5,
		})

		kernel := kernels.NewRBFKernel(1.0, 1.0)
		gp := NewGP(kernel, 1e-6)
		
		// Fit the model
		err := gp.Fit(X, y)
		require.NoError(t, err, "should fit the model")
		
		// Generate samples from the posterior
		rng := rand.New(rand.NewSource(42))
		samples, err := gp.Sample(XTest, 10, rng) // Draw 3 samples
		require.NoError(t, err, "should sample from posterior")
		assert.Equal(t, 10, samples.RawMatrix().Cols, "should return 10 samples")
		assert.Equal(t, nTest, samples.RawMatrix().Rows, "samples should have %d rows", nTest)
		
		// Check that samples are not all zeros
		hasNonZero := false
		for i := 0; i < nTest; i++ {
			for j := 0; j < 3; j++ {
				if samples.At(i, j) != 0 {
					hasNonZero = true
					break
				}
			}
			if hasNonZero {
				break
			}
		}
		assert.True(t, hasNonZero, "samples should have non-zero values")
	})
}

func TestGPConvergence(t *testing.T) {
	t.Skip("This is a long-running test for convergence analysis")
	
	// This test verifies that the GP can learn a simple function
	nTrain := 20
	nTest := 100
	nFeatures := 1
	
	// True function: f(x) = sin(x)
	trueFunc := func(x float64) float64 {
		return math.Sin(x)
	}

	// Generate training data
	X := mat.NewDense(nTrain, nFeatures, nil)
	y := mat.NewVecDense(nTrain, nil)
	for i := 0; i < nTrain; i++ {
		x := float64(i) * 0.3
		X.Set(i, 0, x)
		y.SetVec(i, trueFunc(x) + 0.1*rand.NormFloat64()) // Add some noise
	}

	// Generate test data
	XTest := mat.NewDense(nTest, nFeatures, nil)
	yTrue := mat.NewVecDense(nTest, nil)
	for i := 0; i < nTest; i++ {
		x := float64(i) * 0.06 // Finer grid for testing
		XTest.Set(i, 0, x)
		yTrue.SetVec(i, trueFunc(x))
	}

	// Create and fit GP
	kernel := kernels.NewRBFKernel(1.0, 1.0)
	gp := NewGP(kernel, 0.1) // Higher noise for robustness
	
	err := gp.Fit(X, y)
	require.NoError(t, err, "should fit the model")
	
	// Make predictions
	yPred, std, err := gp.Predict(XTest)
	require.NoError(t, err, "should make predictions")
	
	// Calculate RMSE
	var sumSqErr float64
	for i := 0; i < nTest; i++ {
		err := yPred.AtVec(i) - yTrue.AtVec(i)
		sumSqErr += err * err
	}
	rmse := math.Sqrt(sumSqErr / float64(nTest))
	
	t.Logf("RMSE: %f", rmse)
	assert.Less(t, rmse, 0.2, "RMSE should be less than 0.2")
	
	// Check that uncertainty is reasonable
	meanStd := 0.0
	for i := 0; i < nTest; i++ {
		meanStd += std.AtVec(i)
	}
	meanStd /= float64(nTest)
	
	t.Logf("Mean standard deviation: %f", meanStd)
	assert.Greater(t, meanStd, 0.0, "Mean standard deviation should be positive")
}
