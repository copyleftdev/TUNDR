package optimization

import (
	"math"
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// testObjectiveFunc is a simple quadratic objective function for testing
func testObjectiveFunc(x []float64) (float64, error) {
	sum := 0.0
	for _, v := range x {
		sum += v * v
	}
	return sum, nil
}

// testNoisyObjectiveFunc adds random noise to the objective function
func testNoisyObjectiveFunc(noiseScale float64) ObjectiveFunction {
	return func(x []float64) (float64, error) {
		val, _ := testObjectiveFunc(x)
		return val + noiseScale*(rand.Float64()-0.5), nil
	}
}

// assertFloat64SlicesEqual checks if two float64 slices are approximately equal
func assertFloat64SlicesEqual(t *testing.T, got, want []float64, tol float64) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}

	for i := range got {
		if math.Abs(got[i]-want[i]) > tol {
			t.Fatalf("at index %d: got %v, want %v (tolerance %v)", i, got[i], want[i], tol)
		}
	}
}

// assertMatDimsEqual checks if two matrices have the same dimensions
func assertMatDimsEqual(t *testing.T, got, want mat.Matrix) {
	t.Helper()

	rg, cg := got.Dims()
	rw, cw := want.Dims()

	if rg != rw || cg != cw {
		t.Fatalf("matrix dimensions mismatch: got %dx%d, want %dx%d", rg, cg, rw, cw)
	}
}

// assertMatEqual checks if two matrices are approximately equal
func assertMatEqual(t *testing.T, got, want mat.Matrix, tol float64) {
	t.Helper()

	assertMatDimsEqual(t, got, want)

	r, c := got.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			g := got.At(i, j)
			w := want.At(i, j)
			if math.Abs(g-w) > tol {
				t.Fatalf("at (%d,%d): got %v, want %v (tolerance %v)", i, j, g, w, tol)
			}
		}
	}
}

// generateRandomMatrix generates a random matrix with values in [min, max]
func generateRandomMatrix(rows, cols int, min, max float64) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = min + rand.Float64()*(max-min)
	}
	return mat.NewDense(rows, cols, data)
}

// generateRandomVector generates a random vector with values in [min, max]
func generateRandomVector(size int, min, max float64) *mat.VecDense {
	data := make([]float64, size)
	for i := range data {
		data[i] = min + rand.Float64()*(max-min)
	}
	return mat.NewVecDense(size, data)
}
