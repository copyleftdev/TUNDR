package kernels

import (
	"math"
	"testing"
)

func TestRBFKernel(t *testing.T) {
	tests := []struct {
		name     string
		x1       []float64
		x2       []float64
		ls       float64
		sv       float64
		expected float64
	}{
		{
			name:     "same point",
			x1:       []float64{1.0, 2.0},
			x2:       []float64{1.0, 2.0},
			ls:       1.0,
			sv:       1.0,
			expected: 1.0,
		},
		{
			name:     "different points",
			x1:       []float64{0.0, 0.0},
			x2:       []float64{1.0, 1.0},
			ls:       1.0,
			sv:       1.0,
			expected: math.Exp(-1.0), // exp(-0.5 * (1+1) / 1^2)
		},
		{
			name:     "with different length scale",
			x1:       []float64{0.0, 0.0},
			x2:       []float64{2.0, 2.0},
			ls:       2.0,
			sv:       1.0,
			expected: math.Exp(-1.0), // exp(-0.5 * (2^2 + 2^2) / 2^2)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kernel := NewRBFKernel(tt.ls, tt.sv)
			result := kernel.Eval(tt.x1, tt.x2)

			if math.Abs(result-tt.expected) > 1e-10 {
				t.Errorf("expected %v, got %v", tt.expected, result)
			}

			// Test symmetry
			result2 := kernel.Eval(tt.x2, tt.x1)
			if math.Abs(result-result2) > 1e-10 {
				t.Error("kernel is not symmetric")
			}
		})
	}
}

func TestMatern52Kernel(t *testing.T) {
	tests := []struct {
		name           string
		lengthScale    float64
		signalVariance float64
		x1, x2         []float64
		expected       float64
	}{
		{
			name:           "same point",
			lengthScale:    1.0,
			signalVariance: 1.0,
			x1:             []float64{1.0, 2.0},
			x2:             []float64{1.0, 2.0},
			expected:       1.0,
		},
		{
			name:           "different points",
			lengthScale:    1.0,
			signalVariance: 1.0,
			x1:             []float64{0.0, 0.0},
			x2:             []float64{1.0, 1.0},
			// Expected value calculated manually
			expected:       (1.0 + math.Sqrt(5)*math.Sqrt(2) + (5.0/3.0)*2) * math.Exp(-math.Sqrt(5)*math.Sqrt(2)),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kernel := NewMatern52Kernel(tt.lengthScale, tt.signalVariance)
			result := kernel.Eval(tt.x1, tt.x2)

			if math.Abs(result-tt.expected) > 1e-10 {
				t.Errorf("expected %v, got %v", tt.expected, result)
			}

			// Test symmetry
			result2 := kernel.Eval(tt.x2, tt.x1)
			if math.Abs(result-result2) > 1e-10 {
				t.Error("kernel is not symmetric")
			}
		})
	}
}

func TestKernelHyperparameters(t *testing.T) {
	tests := []struct {
		name     string
		kernel   Kernel
		params   []float64
		wantErr  bool
		errorMsg string
	}{
		{
			name:     "RBF valid params",
			kernel:   NewRBFKernel(1.0, 1.0),
			params:   []float64{2.0, 3.0},
			wantErr:  false,
			errorMsg: "",
		},
		{
			name:     "RBF invalid params count",
			kernel:   NewRBFKernel(1.0, 1.0),
			params:   []float64{1.0},
			wantErr:  true,
			errorMsg: "expected 2 hyperparameters, got 1",
		},
		{
			name:     "RBF invalid param value",
			kernel:   NewRBFKernel(1.0, 1.0),
			params:   []float64{-1.0, 1.0},
			wantErr:  true,
			errorMsg: "hyperparameters must be positive, got [-1 1]",
		},
		{
			name:     "Matern52 valid params",
			kernel:   NewMatern52Kernel(1.0, 1.0),
			params:   []float64{2.0, 3.0},
			wantErr:  false,
			errorMsg: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.kernel.SetHyperparameters(tt.params)

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				if err.Error() != tt.errorMsg {
					t.Errorf("expected error message '%s', got '%s'", tt.errorMsg, err.Error())
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}

				// Verify hyperparameters were set correctly
				params := tt.kernel.Hyperparameters()
				if len(params) != len(tt.params) {
					t.Fatalf("expected %d parameters, got %d", len(tt.params), len(params))
				}
				for i, p := range params {
					if p != tt.params[i] {
						t.Errorf("parameter %d: expected %v, got %v", i, tt.params[i], p)
					}
				}
			}
		})
	}
}
