package bayesian

import (
	"math/rand"
	"runtime"
	"testing"

	"gonum.org/v1/gonum/mat"
	"github.com/tundr/mcp-optimization-server/internal/optimization/kernels"
)

// BenchmarkGPFitScaling measures how GP fitting scales with input size
func BenchmarkGPFitScaling(b *testing.B) {
	tests := []struct {
		name      string
		nSamples  int
		nFeatures int
	}{
		{"Small", 100, 5},
		{"Medium", 1000, 10},
		{"Large", 5000, 20},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			// Generate random data
			X := mat.NewDense(tt.nSamples, tt.nFeatures, nil)
			y := mat.NewVecDense(tt.nSamples, nil)
			for i := 0; i < tt.nSamples; i++ {
				for j := 0; j < tt.nFeatures; j++ {
					X.Set(i, j, rand.NormFloat64())
				}
				y.SetVec(i, rand.NormFloat64())
			}

			// Create GP with Matern52 kernel
			kernel := kernels.NewMatern52Kernel(1.0, 1.0)
			gp := NewGP(kernel, 1e-6)

			// Reset timer and run benchmark
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = gp.Fit(X, y)
			}

			// Report memory allocation
			b.ReportAllocs()
		})
	}
}

// BenchmarkKernelComparison compares performance of different kernel types
func BenchmarkKernelComparison(b *testing.B) {
	tests := []struct {
		name   string
		kernel kernels.Kernel
	}{
		{"RBF", kernels.NewRBFKernel(1.0, 1.0)},
		{"Matern52", kernels.NewMatern52Kernel(1.0, 1.0)},
		// Using Matern52 instead of Matern32 as it's available
		{"Matern52", kernels.NewMatern52Kernel(1.0, 1.0)},
	}

	nSamples := 1000
	nFeatures := 10
	X := mat.NewDense(nSamples, nFeatures, nil)
	y := mat.NewVecDense(nSamples, nil)

	// Generate random data
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, rand.NormFloat64())
		}
		y.SetVec(i, rand.NormFloat64())
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			gp := NewGP(tt.kernel, 1e-6)
			// Warm-up
			if err := gp.Fit(X, y); err != nil {
				b.Fatalf("Failed to fit GP: %v", err)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = gp.Fit(X, y)
			}
		})
	}
}

// BenchmarkGPConcurrent measures performance under concurrent access
func BenchmarkGPConcurrent(b *testing.B) {
	nSamples := 1000
	nFeatures := 10
	nWorkers := runtime.NumCPU()

	tests := []struct {
		name        string
		concurrency int
	}{
		{"Sequential", 1},
		{"Concurrent", nWorkers},
	}

	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			// Create and fit GP
			kernel := kernels.NewMatern52Kernel(1.0, 1.0)
			gp := NewGP(kernel, 1e-6)

			// Generate random data
			X := mat.NewDense(nSamples, nFeatures, nil)
			y := mat.NewVecDense(nSamples, nil)
			for i := 0; i < nSamples; i++ {
				for j := 0; j < nFeatures; j++ {
					X.Set(i, j, rand.NormFloat64())
				}
				y.SetVec(i, rand.NormFloat64())
			}

			// Warm-up
			if err := gp.Fit(X, y); err != nil {
				b.Fatalf("Failed to fit GP: %v", err)
			}

			// Create test points
			nTest := 100
			XTest := mat.NewDense(nTest, nFeatures, nil)
			for i := 0; i < nTest; i++ {
				for j := 0; j < nFeatures; j++ {
					XTest.Set(i, j, rand.NormFloat64()*2-1)
				}
			}

			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					_, _, _ = gp.Predict(XTest)
				}
			})
		})
	}
}

// BenchmarkMemoryUsage measures memory allocations of key operations
func BenchmarkMemoryUsage(b *testing.B) {
	nSamples := 1000
	nFeatures := 10
	nTest := 100

	// Generate random data
	X := mat.NewDense(nSamples, nFeatures, nil)
	y := mat.NewVecDense(nSamples, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nFeatures; j++ {
			X.Set(i, j, rand.NormFloat64())
		}
		y.SetVec(i, rand.NormFloat64())
	}

	// Create test points
	XTest := mat.NewDense(nTest, nFeatures, nil)
	for i := 0; i < nTest; i++ {
		for j := 0; j < nFeatures; j++ {
			XTest.Set(i, j, rand.NormFloat64()*2-1)
		}
	}

	// Test memory usage of GP fitting
	b.Run("GPFit", func(b *testing.B) {
		kernel := kernels.NewMatern52Kernel(1.0, 1.0)
		gp := NewGP(kernel, 1e-6)

		b.ReportAllocs()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			_ = gp.Fit(X, y)
		}
	})

	// Test memory usage of prediction
	b.Run("GPPredict", func(b *testing.B) {
		kernel := kernels.NewMatern52Kernel(1.0, 1.0)
		gp := NewGP(kernel, 1e-6)
		if err := gp.Fit(X, y); err != nil {
			b.Fatalf("Failed to fit GP: %v", err)
		}

		b.ReportAllocs()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			_, _, _ = gp.Predict(XTest)
		}
	})

	// Test memory usage of sampling
	b.Run("GPSample", func(b *testing.B) {
		kernel := kernels.NewMatern52Kernel(1.0, 1.0)
		gp := NewGP(kernel, 1e-6)
		if err := gp.Fit(X, y); err != nil {
			b.Fatalf("Failed to fit GP: %v", err)
		}

		rng := rand.New(rand.NewSource(42))
		b.ReportAllocs()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			_, _ = gp.Sample(XTest, 10, rng)
		}
	})
}
