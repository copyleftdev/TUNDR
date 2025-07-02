package kernels

import (
	"fmt"
	"math"
)

// Kernel represents a kernel function for Gaussian Processes
type Kernel interface {
	// Eval computes the kernel value between two points x1 and x2
	Eval(x1, x2 []float64) float64

	// Hyperparameters returns the current hyperparameters
	Hyperparameters() []float64

	// SetHyperparameters sets the kernel's hyperparameters
	SetHyperparameters(params []float64) error
}

// RBFKernel implements the Radial Basis Function (squared exponential) kernel
type RBFKernel struct {
	// Length scale parameter (larger = smoother function)
	lengthScale float64
	// Signal variance (controls the amplitude of the function)
	signalVar float64
}

// NewRBFKernel creates a new RBF kernel with the given parameters
func NewRBFKernel(lengthScale, signalVar float64) *RBFKernel {
	if lengthScale <= 0 {
		panic(fmt.Sprintf("lengthScale must be positive, got %v", lengthScale))
	}
	if signalVar <= 0 {
		panic(fmt.Sprintf("signalVar must be positive, got %v", signalVar))
	}
	return &RBFKernel{
		lengthScale: lengthScale,
		signalVar:   signalVar,
	}
}

// Eval computes the RBF kernel value between x1 and x2
func (k *RBFKernel) Eval(x1, x2 []float64) float64 {
	sumSq := 0.0
	for i := range x1 {
		diff := x1[i] - x2[i]
		sumSq += diff * diff
	}
	r2 := sumSq / (2.0 * k.lengthScale * k.lengthScale)
	return k.signalVar * math.Exp(-r2)
}

// Hyperparameters returns the current hyperparameters
func (k *RBFKernel) Hyperparameters() []float64 {
	return []float64{k.lengthScale, k.signalVar}
}

// SetHyperparameters sets the kernel's hyperparameters
func (k *RBFKernel) SetHyperparameters(params []float64) error {
	if len(params) != 2 {
		return fmt.Errorf("expected 2 hyperparameters, got %d", len(params))
	}
	if params[0] <= 0 || params[1] <= 0 {
		return fmt.Errorf("hyperparameters must be positive, got %v", params)
	}
	k.lengthScale = params[0]
	k.signalVar = params[1]
	return nil
}

// Matern52Kernel implements the Matérn 5/2 kernel
type Matern52Kernel struct {
	// Length scale parameter (larger = smoother function)
	lengthScale float64
	// Signal variance (controls the amplitude of the function)
	signalVar float64
}

// NewMatern52Kernel creates a new Matérn 5/2 kernel with the given parameters
func NewMatern52Kernel(lengthScale, signalVar float64) *Matern52Kernel {
	if lengthScale <= 0 {
		panic(fmt.Sprintf("lengthScale must be positive, got %v", lengthScale))
	}
	if signalVar <= 0 {
		panic(fmt.Sprintf("signalVar must be positive, got %v", signalVar))
	}
	return &Matern52Kernel{
		lengthScale: lengthScale,
		signalVar:   signalVar,
	}
}

// Eval computes the Matérn 5/2 kernel value between x1 and x2
func (k *Matern52Kernel) Eval(x1, x2 []float64) float64 {
	sumSq := 0.0
	for i := range x1 {
		diff := x1[i] - x2[i]
		sumSq += diff * diff
	}
	r := math.Sqrt(sumSq) / k.lengthScale
	polyTerm := 1.0 + math.Sqrt(5)*r + (5.0/3.0)*r*r
	expTerm := math.Exp(-math.Sqrt(5)*r)
	return k.signalVar * polyTerm * expTerm
}

// Hyperparameters returns the current hyperparameters
func (k *Matern52Kernel) Hyperparameters() []float64 {
	return []float64{k.lengthScale, k.signalVar}
}

// SetHyperparameters sets the kernel's hyperparameters
func (k *Matern52Kernel) SetHyperparameters(params []float64) error {
	if len(params) != 2 {
		return fmt.Errorf("expected 2 hyperparameters, got %d", len(params))
	}
	if params[0] <= 0 || params[1] <= 0 {
		return fmt.Errorf("hyperparameters must be positive, got %v", params)
	}
	k.lengthScale = params[0]
	k.signalVar = params[1]
	return nil
}
