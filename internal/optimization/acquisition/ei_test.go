package acquisition

import (
	"math"
	"testing"
)

func TestExpectedImprovement(t *testing.T) {
	tests := []struct {
		name          string
		bestObserved  float64
		xi            float64
		mu            float64
		sigma         float64
		expectedValue float64
	}{
		{
			name:          "no improvement",
			bestObserved:  1.0, // Best observed value is 1.0
			xi:            0.01,
			mu:            1.5,  // Current point is worse (1.5 > 1.0)
			sigma:         0.1,
			expectedValue: 0.0,  // No improvement expected
		},
		{
			name:          "definite improvement",
			bestObserved:  1.0, // Best observed value is 1.0
			xi:            0.01,
			mu:            0.5,  // Current point is better (0.5 < 1.0)
			sigma:         0.2,
			expectedValue: 0.4905, // Expected EI value for this case (1.0 - 0.5 - 0.01 = 0.49), but actual includes some PDF contribution
		},
		{
			name:          "zero sigma",
			bestObserved:  1.0, // Best observed value is 1.0
			xi:            0.0,
			mu:            0.5,  // Current point is better (0.5 < 1.0)
			sigma:         0.0,
			expectedValue: 0.5,  // bestObserved - mu - xi = 1.0 - 0.5 - 0.0 = 0.5
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ei := NewExpectedImprovement(tt.bestObserved, tt.xi)
			// Set to minimization mode since that's what the implementation expects
			ei.minimize = true
			result := ei.Compute(tt.mu, tt.sigma)

			// Use a small tolerance for floating point comparison
			tolerance := 1e-4 // Increased tolerance for floating point comparisons
			if math.Abs(result-tt.expectedValue) > tolerance {
				t.Errorf("expected %v, got %v (tolerance %v)", tt.expectedValue, result, tolerance)
			}
		})
	}
}

func TestExpectedImprovementUpdate(t *testing.T) {
	ei := NewExpectedImprovement(1.0, 0.01)
	ei.minimize = true // Set to minimization mode

	// Initial best should be 1.0
	if ei.BestObserved() != 1.0 {
		t.Errorf("initial best observed should be 1.0, got %v", ei.BestObserved())
	}

	// Update best to a better value (lower is better for minimization)
	ei.UpdateBest(0.5)
	if ei.BestObserved() != 0.5 {
		t.Errorf("updated best observed should be 0.5, got %v", ei.BestObserved())
	}

	// Update xi with a smaller value to avoid the edge case where improvement is exactly 0.0
	ei.SetXi(0.01)
	// Test with a point that's better than the current best (0.4 < 0.5)
	result := ei.Compute(0.4, 0.1)
	if result <= 0 {
		t.Error("expected positive EI after update")
	}
}

func TestExpectedImprovementGradient(t *testing.T) {
	tests := []struct {
		name          string
		bestObserved  float64
		xi            float64
		mu            float64
		sigma         float64
		dmu           float64
		dsigma        float64
		h             float64 // Step size for finite differences
		tolerance     float64 // Allowed error
		minimize      bool    // Whether to minimize or maximize
	}{
		{
			name:         "gradient test 1",
			bestObserved: 1.0,
			xi:           0.01,
			mu:           0.5,
			sigma:        0.5,
			dmu:          1.0,
			dsigma:       1.0,
			h:            1e-6,
			tolerance:    1e-6,
			minimize:     false, // Test with maximization
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ei := NewExpectedImprovement(tt.bestObserved, tt.xi)
			
			// Compute gradient using the method
			grad := ei.Gradient(tt.mu, tt.dmu, tt.sigma, tt.dsigma)
			
			// Compute numerical gradient using central differences
			f := func(eps float64) float64 {
				return ei.Compute(tt.mu+eps*tt.dmu, tt.sigma+eps*tt.dsigma)
			}
			
			numericalGrad := (f(tt.h) - f(-tt.h)) / (2 * tt.h)
			
			// Compare
			if math.Abs(grad-numericalGrad) > tt.tolerance {
				t.Errorf("gradient mismatch: got %v, want %v (tolerance %v)", 
					grad, numericalGrad, tt.tolerance)
			}
		})
	}
}
