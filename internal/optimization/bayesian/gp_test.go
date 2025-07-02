package bayesian

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gonum.org/v1/gonum/mat"

	"github.com/copyleftdev/TUNDR/internal/optimization/kernels"
)

func TestGPFitAndPredict(t *testing.T) {
	// Simple test with 3 points
	X := mat.NewDense(3, 1, []float64{1, 2, 3})
	y := mat.NewVecDense(3, []float64{1, 2, 1})

	gp := NewGP(kernels.NewRBFKernel(1.0, 1.0), 1e-6)
	err := gp.Fit(X, y)
	assert.NoError(t, err)

	// Test prediction at training points
	Xtest := mat.NewDense(3, 1, []float64{1, 2, 3})
	mean, var_, err := gp.Predict(Xtest)
	assert.NoError(t, err)
	assert.NotNil(t, mean)
	assert.NotNil(t, var_)
}

func TestGPSampling(t *testing.T) {
	// Create some test data
	X := mat.NewDense(3, 1, []float64{1, 2, 3})
	y := mat.NewVecDense(3, []float64{1, 2, 1})

	// Create and fit GP
	gp := NewGP(kernels.NewRBFKernel(1.0, 1.0), 1e-6)
	err := gp.Fit(X, y)
	require.NoError(t, err)

	// Generate samples with a fixed seed for reproducibility
	rng := rand.New(rand.NewSource(42))
	samples, err := gp.Sample(X, 5, rng)
	require.NoError(t, err)

	// Check dimensions - in the current implementation:
	// - Rows = number of points
	// - Cols = number of samples
	nPoints, nSamples := samples.Dims()
	assert.Equal(t, 3, nPoints, "number of points should match input dimensions")
	assert.Equal(t, 5, nSamples, "number of samples should match")

	// Check that samples are different
	for i := 1; i < nSamples; i++ {
		same := true
		for j := 0; j < nPoints; j++ {
			if samples.At(j, i) != samples.At(j, 0) {
				same = false
				break
			}
		}
		assert.False(t, same, "samples should be different")
	}
}

func TestGPWithNoise(t *testing.T) {
	// Test that noise is handled correctly
	X := mat.NewDense(3, 1, []float64{-1, 0, 1})
	y := mat.NewVecDense(3, []float64{1, 0, 1})

	// Create GP with significant noise
	kernel := kernels.NewRBFKernel(1.0, 1.0)
	gp := NewGP(kernel, 0.1) // Larger noise

	err := gp.Fit(X, y)
	require.NoError(t, err)

	// Predict at training points - should not interpolate exactly due to noise
	testX := mat.NewDense(3, 1, []float64{-1, 0, 1})
	means, variances, err := gp.Predict(testX)
	require.NoError(t, err)

	// Check that predictions are close but not exact
	for i := 0; i < 3; i++ {
		assert.InDelta(t, y.AtVec(i), means.AtVec(i), 0.5, "prediction should be close to training data")
		assert.Greater(t, variances.AtVec(i), 0.0, "variance should be positive")
	}
}

func TestGPErrorHandling(t *testing.T) {
	// Test error cases
	kernel := kernels.NewRBFKernel(1.0, 1.0)
	gp := NewGP(kernel, 1e-6)

	// Test empty input
	t.Run("empty input", func(t *testing.T) {
		// Create empty inputs without using mat.NewDense/NewVecDense to avoid panic
		var emptyX *mat.Dense
		var emptyY *mat.VecDense
		
		err := gp.Fit(emptyX, emptyY)
		require.Error(t, err, "should error on nil input")
		assert.Contains(t, err.Error(), "input matrices must not be nil", "error should indicate nil input")
		
		// Test with zero-length but non-nil inputs
		emptyX = &mat.Dense{}
		emptyY = &mat.VecDense{}
		err = gp.Fit(emptyX, emptyY)
		require.Error(t, err, "should error on zero-length input")
		assert.Contains(t, err.Error(), "input matrix X must not be empty", "error should indicate empty input")
	})

	t.Run("mismatched dimensions", func(t *testing.T) {
		X := mat.NewDense(3, 1, []float64{1, 2, 3})
		y := mat.NewVecDense(2, []float64{1, 2}) // Wrong length
		err := gp.Fit(X, y)
		require.Error(t, err, "should error on mismatched dimensions")
		assert.Contains(t, err.Error(), "dimension mismatch: X has 3 samples but y has length 2", "error should indicate dimension mismatch")
	})

	t.Run("predict without fit", func(t *testing.T) {
		_, _, err := gp.Predict(mat.NewDense(1, 1, []float64{0}))
		require.Error(t, err, "should error when predicting without fitting")
		assert.Contains(t, err.Error(), "model not trained or no training data", "error should indicate model not fitted")
	})

	t.Run("sample without fit", func(t *testing.T) {
		rng := rand.New(rand.NewSource(42))
		_, err := gp.Sample(mat.NewDense(1, 1, []float64{0}), 1, rng)
		require.Error(t, err, "should error when sampling without fitting")
		assert.Contains(t, err.Error(), "model not trained or no training data", "error should indicate model not fitted")
	})
}

func TestGPSingularMatrix(t *testing.T) {
	// Test handling of singular matrix (duplicate points)
	X := mat.NewDense(3, 1, []float64{1.0, 1.0, 1.0}) // All points the same
	y := mat.NewVecDense(3, []float64{1.0, 1.0, 1.1})  // Slightly different y values

	kernel := kernels.NewRBFKernel(1.0, 1.0)
	gp := NewGP(kernel, 1e-6)

	// This should add jitter and succeed
	err := gp.Fit(X, y)
	require.NoError(t, err)

	// Should be able to make predictions
	testX := mat.NewDense(1, 1, []float64{1.0})
	_, variances, err := gp.Predict(testX)
	require.NoError(t, err)
	assert.Greater(t, variances.AtVec(0), 0.0, "should have positive variance")
}

func TestGPBatchPredict(t *testing.T) {
	// Test batch prediction
	X := mat.NewDense(5, 1, []float64{-2, -1, 0, 1, 2})
	y := mat.NewVecDense(5, []float64{4, 1, 0, 1, 4}) // x^2

	kernel := kernels.NewRBFKernel(1.0, 1.0)
	gp := NewGP(kernel, 1e-6)

	err := gp.Fit(X, y)
	require.NoError(t, err)

	// Test points
	testX := mat.NewDense(3, 1, []float64{-0.5, 0.5, 1.5})
	means, variances, err := gp.Predict(testX)
	require.NoError(t, err)

	// Check dimensions
	nPoints, _ := testX.Dims()
	assert.Equal(t, nPoints, means.Len(), "means length should match number of test points")
	assert.Equal(t, nPoints, variances.Len(), "variances length should match number of test points")

	// Check predictions are reasonable
	for i := 0; i < nPoints; i++ {
		x := testX.At(i, 0)
		expectedY := x * x
		assert.InDelta(t, expectedY, means.AtVec(i), 0.5, "prediction should be close to x^2")
		assert.Greater(t, variances.AtVec(i), 0.0, "variance should be positive")
	}
}
