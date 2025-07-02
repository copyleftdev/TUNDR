package acquisition

import (
	"fmt"

	"gonum.org/v1/gonum/stat/distuv"
)

// ExpectedImprovement implements the Expected Improvement acquisition function
type ExpectedImprovement struct {
	// Best observed value so far
	bestObserved float64
	// Exploration-exploitation trade-off parameter (xi)
	xi float64
	// Whether we're minimizing (true) or maximizing (false)
	minimize bool
}

// NewExpectedImprovement creates a new ExpectedImprovement acquisition function
// By default, it assumes we're minimizing (lower values are better)
func NewExpectedImprovement(bestObserved, xi float64) *ExpectedImprovement {
	return &ExpectedImprovement{
		bestObserved: bestObserved,
		xi:           xi,
		minimize:     true, // Default to minimization
	}
}

// Compute computes the Expected Improvement at point x
// mu: mean prediction at x
// sigma: standard deviation of prediction at x
// Returns the expected improvement value (always non-negative)
func (ei *ExpectedImprovement) Compute(mu, sigma float64) float64 {
	// Debug logging for input parameters
	fmt.Printf("\n=== Compute EI ===\n")
	fmt.Printf("bestObserved=%.4f, mu=%.4f, sigma=%.4f, xi=%.4f, minimize=%v\n",
		ei.bestObserved, mu, sigma, ei.xi, ei.minimize)

	var improvement float64

	if ei.minimize {
		// For minimization, we want to find points with lower function values
		// improvement = best_observed - mu - xi
		improvement = ei.bestObserved - mu - ei.xi
		fmt.Printf("Minimization: improvement = bestObserved - mu - xi = %.4f - %.4f - %.4f = %.4f\n",
			ei.bestObserved, mu, ei.xi, improvement)
	} else {
		// For maximization, we want to find points with higher function values
		// improvement = mu - best_observed - xi
		improvement = mu - ei.bestObserved - ei.xi
		fmt.Printf("Maximization: improvement = mu - bestObserved - xi = %.4f - %.4f - %.4f = %.4f\n",
			mu, ei.bestObserved, ei.xi, improvement)
	}

	// If no improvement is possible, return 0
	if improvement <= 0 {
		fmt.Printf("No improvement possible (improvement=%.4f <= 0), returning 0\n", improvement)
		return 0.0
	}

	// If sigma is zero or very small, return the improvement directly
	// This handles the case where we're certain about the prediction
	if sigma <= 1e-10 {
		fmt.Printf("Sigma is zero or very small (%.4f), returning improvement=%.4f\n", sigma, improvement)
		return improvement
	}

	// Standard normal distribution
	stdNormal := distuv.UnitNormal
	z := improvement / sigma
	fmt.Printf("z = improvement / sigma = %.4f / %.4f = %.4f\n", improvement, sigma, z)

	// Calculate EI using the standard formula:
	// EI = improvement * Φ(z) + sigma * φ(z)
	// where Φ is the CDF and φ is the PDF of the standard normal distribution
	pdf := stdNormal.Prob(z)
	cdf := stdNormal.CDF(z)
	fmt.Printf("PDF(%.4f) = %.4f, CDF(%.4f) = %.4f\n", z, pdf, z, cdf)

	eiValue := improvement*cdf + sigma*pdf
	fmt.Printf("EI = improvement * CDF + sigma * PDF = %.4f * %.4f + %.4f * %.4f = %.4f\n", 
		improvement, cdf, sigma, pdf, eiValue)

	return eiValue
}

// Gradient computes the gradient of the Expected Improvement
// dmu: derivative of mu with respect to the parameter
// dsigma: derivative of sigma with respect to the parameter
func (ei *ExpectedImprovement) Gradient(mu, dmu float64, sigma, dsigma float64) float64 {
	if sigma <= 1e-10 {
		// When sigma is very small, EI is linear in mu
		if ei.minimize {
			return -dmu // EI = best_observed - mu - xi
		}
		return dmu // EI = mu - best_observed - xi
	}

	var improvement float64
	if ei.minimize {
		improvement = ei.bestObserved - mu - ei.xi
	} else {
		improvement = mu - ei.bestObserved - ei.xi
	}

	if improvement <= 0 {
		return 0.0
	}

	z := improvement / sigma
	stdNormal := distuv.UnitNormal
	pdf := stdNormal.Prob(z)
	cdf := stdNormal.CDF(z)

	// The gradient of EI with respect to mu is -cdf for minimization, +cdf for maximization
	// The gradient of EI with respect to sigma is always pdf
	// Total gradient is (dEI/dmu)*dmu + (dEI/dsigma)*dsigma
	if ei.minimize {
		return -cdf*dmu + pdf*dsigma
	}
	return cdf*dmu + pdf*dsigma
}

// UpdateBest updates the best observed value
func (ei *ExpectedImprovement) UpdateBest(best float64) {
	ei.bestObserved = best
}

// SetXi sets the exploration-exploitation trade-off parameter
func (ei *ExpectedImprovement) SetXi(xi float64) {
	ei.xi = xi
}

// BestObserved returns the best observed value
func (ei *ExpectedImprovement) BestObserved() float64 {
	return ei.bestObserved
}
