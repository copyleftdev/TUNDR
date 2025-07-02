package bayesian

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"go.uber.org/zap"
	"gonum.org/v1/gonum/mat"
	"github.com/copyleftdev/TUNDR/internal/optimization"
	"github.com/copyleftdev/TUNDR/internal/optimization/kernels"
)

// GP implements a Gaussian Process model for Bayesian Optimization
// GP implements a Gaussian Process model for Bayesian Optimization
type GP struct {
	// Kernel function
	kernel kernels.Kernel

	// Mean function
	meanFunc func([]float64) float64

	// Noise variance
	noiseVar float64

	// Training data
	X *mat.Dense // Input points (n_samples, n_features)
	y *mat.VecDense // Target values (n_samples)

	// Precomputed values
	alpha *mat.VecDense
	L     *mat.Cholesky
	
	// Matrix pool for reusing matrix allocations
	matrixPool *MatrixPool

	// Logger for structured logging
	logger *zap.Logger
}

// NewGP creates a new Gaussian Process model
func NewGP(kernel kernels.Kernel, noiseVar float64) *GP {
	// Create a logger with default settings (zap's development config for now)
	logger, _ := zap.NewDevelopment()

	return &GP{
		kernel:     kernel,
		noiseVar:   noiseVar,
		meanFunc:   zeroMean,
		matrixPool: NewMatrixPool(),
		logger:     logger.Named("gaussian_process"),
	}
}

// Fit fits the GP model to the training data
func (gp *GP) Fit(X *mat.Dense, y *mat.VecDense) error {
	const op = "GP.Fit"
	
	if X == nil || y == nil {
		err := errors.New("input matrices must not be nil")
		return optimization.WrapError(err, "gaussian_process: "+op)
	}

	nSamples, nFeatures := X.Dims()
	yLen := y.Len()

	if nSamples == 0 || nFeatures == 0 {
		err := errors.New("input matrix X must not be empty")
		return optimization.WrapError(err, "gaussian_process: "+op)
	}

	if nSamples != yLen {
		err := fmt.Errorf("dimension mismatch: X has %d samples but y has length %d", 
			nSamples, yLen)
		return optimization.WrapError(err, "gaussian_process: "+op)
	}

	gp.logger.Debug("Fitting GP model",
		zap.Int("samples", nSamples),
		zap.Int("features", nFeatures),
		zap.Float64("noise_var", gp.noiseVar),
	)

	// Store training data
	gp.X = mat.DenseCopyOf(X)
	gp.y = mat.VecDenseCopyOf(y)

	// Compute kernel matrix
	K, err := gp.computeKernelMatrix(X, nSamples)
	if err != nil {
		return optimization.WrapError(fmt.Errorf("failed to compute kernel matrix: %w", err), "gaussian_process: "+op)
	}

	// Add noise to the diagonal
	for i := 0; i < nSamples; i++ {
		K.SetSym(i, i, K.At(i, i)+gp.noiseVar)
	}

	// Solve for alpha: K * alpha = y
	alpha, err := gp.solveLinearSystem(K, y, nSamples)
	if err != nil {
		return optimization.WrapError(fmt.Errorf("failed to solve linear system: %w", err), "gaussian_process: "+op)
	}

	// Store solution
	gp.alpha = alpha

	// Compute Cholesky decomposition for predictions
	var chol mat.Cholesky
	if ok := chol.Factorize(K); !ok {
		err := errors.New("Cholesky decomposition failed: matrix is not positive definite")
		return optimization.WrapError(err, "gaussian_process: "+op)
	}
	gp.L = &chol

	gp.logger.Debug("Successfully fitted GP model",
		zap.Int("samples", nSamples),
		zap.Int("features", nFeatures),
	)

	return nil
}

// computeKernelMatrix computes the kernel matrix with improved numerical stability
func (gp *GP) computeKernelMatrix(X *mat.Dense, nSamples int) (*mat.SymDense, error) {
	const op = "GP.computeKernelMatrix"
	
	if X == nil || nSamples <= 0 {
		err := errors.New("invalid input to computeKernelMatrix")
		return nil, optimization.WrapError(err, "gaussian_process: "+op)
	}

	gp.logger.Debug("Computing kernel matrix",
		zap.Int("nSamples", nSamples),
		zap.Float64("noiseVar", gp.noiseVar),
	)

	// Get a symmetric matrix from the pool or create a new one
	K := gp.matrixPool.GetSymDense(nSamples)
	
	// First pass: compute all kernel values and find max diagonal
	maxDiag := 0.0
	for i := 0; i < nSamples; i++ {
		x1 := mat.Row(nil, i, X)
		diag := gp.kernel.Eval(x1, x1)
		maxDiag = math.Max(maxDiag, diag)
	}
	
	// If max diagonal is zero, use a small default to avoid division by zero
	if maxDiag <= 0 {
		maxDiag = 1.0
	}

	// Second pass: compute all kernel values with scaling
	for i := 0; i < nSamples; i++ {
		x1 := mat.Row(nil, i, X)
		// Diagonal element with scaling
		diag := gp.kernel.Eval(x1, x1) / maxDiag
		K.SetSym(i, i, diag)
		
		// Off-diagonal elements with scaling
		for j := i + 1; j < nSamples; j++ {
			x2 := mat.Row(nil, j, X)
			k := gp.kernel.Eval(x1, x2) / maxDiag
			K.SetSym(i, j, k)
		}
	}

	// Compute the trace for scaling
	trace := mat.Trace(K)
	if trace <= 0 {
		trace = 1.0
	}

	// Add noise and jitter to diagonal with better scaling
	diag := make([]float64, nSamples)
	for i := 0; i < nSamples; i++ {
		diag[i] = K.At(i, i)
	}

	for i := 0; i < nSamples; i++ {
		current := diag[i]
		// Add noise variance and a small jitter for numerical stability
		// Scale jitter relative to the magnitude of the diagonal element
		jitter := 1e-12 * math.Abs(current)
		if jitter < 1e-12 {
			jitter = 1e-12
		}
		// Ensure we don't make the diagonal smaller than it already is
		adjustment := math.Max(0, gp.noiseVar + jitter - current)
		newVal := current + adjustment
		K.SetSym(i, i, newVal)
		
		// Log if we had to adjust the diagonal significantly
		if adjustment > 0 && math.Abs(adjustment) > 1e-8 * math.Abs(current) {
			gp.logger.Debug("Adjusted diagonal element for numerical stability",
				zap.Int("i", i),
				zap.Float64("original", current),
				zap.Float64("adjustment", adjustment),
				zap.Float64("new_value", newVal))
		}
	}

	// Log the condition number of the matrix
	var svd mat.SVD
	if svd.Factorize(K, mat.SVDNone) {
		s := svd.Values(nil)
		if len(s) > 0 {
			var cond float64
			if s[len(s)-1] > 0 {
				cond = s[0] / s[len(s)-1]
			} else {
				cond = math.Inf(1)
			}
			gp.logger.Debug("Kernel matrix condition number",
				zap.Float64("condition_number", cond),
				zap.Float64("max_singular_value", s[0]),
				zap.Float64("min_singular_value", s[len(s)-1]),
			)
		}
	}

	return K, nil
}

// Predict returns the mean and variance of the posterior predictive distribution
// at the given test points X*. The mean and variance are returned as *mat.VecDense
// and *mat.VecDense respectively.
func (gp *GP) Predict(X *mat.Dense) (*mat.VecDense, *mat.VecDense, error) {
	const op = "GP.Predict"
	
	if X == nil {
		return nil, nil, optimization.WrapError(
			errors.New("input matrix X is nil"), 
			"gaussian_process: "+op,
		)
	}

	if gp == nil || gp.X == nil || gp.alpha == nil {
		return nil, nil, optimization.WrapError(
			errors.New("model not trained or no training data"),
			"gaussian_process: "+op,
		)
	}

	nTest, _ := X.Dims()
	nTrain, nFeatures := gp.X.Dims()
	if nTrain == 0 || nFeatures == 0 {
		return nil, nil, optimization.WrapError(
			errors.New("training data is empty"),
			"gaussian_process: "+op,
		)
	}

	// Allocate output vectors
	mean := mat.NewVecDense(nTest, nil)
	variance := mat.NewVecDense(nTest, nil)

	// Compute kernel matrix between test and training points
	Kss := make([]float64, nTest)
	Kstar := mat.NewDense(nTest, nTrain, nil)

	for i := 0; i < nTest; i++ {
		xStar := X.RawRowView(i)
		Kss[i] = gp.kernel.Eval(xStar, xStar) + gp.noiseVar

		for j := 0; j < nTrain; j++ {
			xTrain := gp.X.RawRowView(j)
			Kstar.Set(i, j, gp.kernel.Eval(xStar, xTrain))
		}
	}

	// Compute mean: K* * alpha
	mean.MulVec(Kstar, gp.alpha)

	// Compute variance: diag(K** - K* * K^-1 * K*^T)
	// We can compute this efficiently using the Cholesky factor
	if gp.L != nil {
		// Solve K * v = K*^T for v
		v := mat.NewDense(nTrain, nTest, nil)
		if err := v.Solve(gp.L, Kstar.T()); err != nil {
			return nil, nil, optimization.WrapError(
				fmt.Errorf("failed to solve linear system: %v", err),
				"gaussian_process: "+op,
			)
		}

		// Compute variance: Kss - sum(v^2, 1)
		for i := 0; i < nTest; i++ {
			var sum float64
			for j := 0; j < nTrain; j++ {
				val := v.At(j, i)
				sum += val * val
			}
			variance.SetVec(i, math.Max(0, Kss[i]-sum)) // Ensure non-negative variance
		}
	}

	// Ensure variance is non-negative (handle numerical issues)
	for i := 0; i < nTest; i++ {
		v := variance.AtVec(i)
		if v < 0 {
			gp.logger.Warn("Negative variance detected, clamping to zero",
				zap.Float64("variance", v),
				zap.Int("test_point", i),
			)
			variance.SetVec(i, 0)
		}
	}

	return mean, variance, nil
}

// Sample draws nSamples samples from the posterior Gaussian Process at the given test points X.
// It returns a matrix where each column represents a sample.
func (gp *GP) Sample(X *mat.Dense, nSamples int, rng *rand.Rand) (*mat.Dense, error) {
	const op = "GP.Sample"

	if X == nil {
		return nil, optimization.WrapError(
			errors.New("input matrix X is nil"),
			"gaussian_process: "+op,
		)
	}

	nTest, _ := X.Dims()
	if nSamples <= 0 {
		return nil, optimization.WrapError(
			errors.New("number of samples must be positive"),
			"gaussian_process: "+op,
		)
	}

	// Get the predictive mean and covariance
	mean, cov, err := gp.Predict(X)
	if err != nil {
		return nil, optimization.WrapError(err, "gaussian_process: "+op)
	}

	// Create output matrix (nTest x nSamples)
	samples := mat.NewDense(nTest, nSamples, nil)

	// Generate standard normal samples (nSamples x nTest)
	stdNorm := mat.NewDense(nSamples, nTest, nil)
	for i := 0; i < nSamples; i++ {
		for j := 0; j < nTest; j++ {
			stdNorm.Set(i, j, rng.NormFloat64())
		}
	}

	// Convert variance vector to diagonal covariance matrix
	n := cov.Len()
	covDense := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		v := cov.AtVec(i)
		if v < 0 {
			v = 0 // Ensure non-negative variance
		}
		covDense.SetSym(i, i, v)
	}

	// Compute Cholesky decomposition of the covariance matrix
	var chol mat.Cholesky
	if ok := chol.Factorize(covDense); !ok {
		// Fall back to SVD with a small tolerance
		var svd mat.SVD
		ok := svd.Factorize(covDense, mat.SVDFull)
		if !ok {
			return nil, optimization.WrapError(
				errors.New("SVD factorization failed"),
				"gaussian_process: "+op,
			)
		}

		var U, V mat.Dense
		svd.UTo(&U)
		svd.VTo(&V)

		s := svd.Values(nil)
		sqrtS := mat.NewDiagDense(len(s), nil)
		for i, val := range s {
			sqrtS.SetDiag(i, math.Sqrt(math.Max(0, val)))
		}

		// Compute sqrt(cov) = U * sqrt(S) * V^T
		var tmp, sqrtCov mat.Dense
		tmp.Mul(&U, sqrtS)
		sqrtCov.Mul(&tmp, V.T())

		// Generate samples: mean + (sqrt(cov) * std_normal^T)^T
		var samplesCentered mat.Dense
		samplesCentered.Mul(stdNorm, &sqrtCov) // (nSamples x nTest) * (nTest x nTest) = nSamples x nTest

		// Transpose and add mean to each sample
		for i := 0; i < nTest; i++ {
			for j := 0; j < nSamples; j++ {
				samples.Set(i, j, mean.AtVec(i) + samplesCentered.At(j, i))
			}
		}

	} else {
		// Generate samples using Cholesky factor
		var L mat.TriDense
		chol.LTo(&L)

		// Generate samples: mean + (L * std_normal^T)^T
		var samplesCentered mat.Dense
		samplesCentered.Mul(stdNorm, &L) // (nSamples x nTest) * (nTest x nTest) = nSamples x nTest

		// Transpose and add mean to each sample
		for i := 0; i < nTest; i++ {
			for j := 0; j < nSamples; j++ {
				samples.Set(i, j, mean.AtVec(i) + samplesCentered.At(j, i))
			}
		}
	}

	return samples, nil
}
func (gp *GP) solveLinearSystem(K *mat.SymDense, y *mat.VecDense, nSamples int) (*mat.VecDense, error) {
	const op = "GP.solveLinearSystem"

	n := y.Len()
	if n == 0 {
		err := errors.New("empty input vector")
		return nil, optimization.WrapError(err, "gaussian_process: "+op)
	}

	// Create a copy of the kernel matrix
	Kcopy := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			Kcopy.SetSym(i, j, K.At(i, j))
		}
	}

	yCopy := mat.NewVecDense(n, nil)
	yCopy.CopyVec(y)

	bestAlpha := mat.NewVecDense(n, nil)
	bestResidual := math.MaxFloat64
	bestJitter := 1e-12

	// Try multiple jitter values if needed
	jitter := bestJitter
	maxAttempts := 10

	for attempt := 0; attempt < maxAttempts; attempt++ {
		// Create a copy of K with jitter added to diagonal
		Kjittered := mat.NewSymDense(n, nil)
		for i := 0; i < n; i++ {
			for j := i; j < n; j++ {
				val := Kcopy.At(i, j)
				if i == j {
					val += jitter
				}
				Kjittered.SetSym(i, j, val)
			}
		}

		// Try Cholesky decomposition
		var chol mat.Cholesky
		if ok := chol.Factorize(Kjittered); !ok {
			gp.logger.Debug("Cholesky factorization failed, increasing jitter",
				zap.Int("attempt", attempt+1),
				zap.Float64("jitter", jitter))
			jitter *= 10
			continue
		}

		// Solve the system using Cholesky
		alpha := mat.NewVecDense(n, nil)
		if err := chol.SolveVecTo(alpha, yCopy); err != nil {
			gp.logger.Debug("Cholesky solve failed, trying SVD",
				zap.Error(err),
				zap.Int("attempt", attempt+1))
			jitter *= 10
			continue
		}

		// Compute residual
		r := mat.NewVecDense(n, nil)
		Kalpha := mat.NewVecDense(n, nil)
		Kalpha.MulVec(Kjittered, alpha)
		r.SubVec(Kalpha, yCopy)
		residual := mat.Norm(r, 2)

		// Update best solution if this one is better
		if residual < bestResidual {
			bestResidual = residual
			bestJitter = jitter
			bestAlpha.CopyVec(alpha)
		}

		// If residual is small enough, we're done
		if residual < 1e-8 {
			gp.logger.Debug("Converged with small residual",
				zap.Float64("residual", residual),
				zap.Float64("jitter", jitter))
			return bestAlpha, nil
		}

		// Increase jitter for next attempt
		jitter *= 10
	}

	// If we get here, Cholesky failed or didn't converge, try SVD as last resort
	gp.logger.Debug("Falling back to SVD after Cholesky attempts failed",
		zap.Float64("best_residual", bestResidual),
		zap.Float64("best_jitter", bestJitter))

	// Convert to dense for SVD
	Kdense := mat.NewDense(n, n, nil)
	yDense := mat.NewDense(n, 1, nil)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			Kdense.Set(i, j, Kcopy.At(i, j))
		}
		yDense.Set(i, 0, yCopy.AtVec(i))
	}

	return gp.solveWithSVD(Kdense, yDense, nSamples)
}

func (gp *GP) solveWithSVD(K, y *mat.Dense, nSamples int) (*mat.VecDense, error) {
	const op = "GP.solveWithSVD"

	if K == nil || y == nil || nSamples == 0 {
		err := errors.New("invalid input to solveWithSVD")
		return nil, optimization.WrapError(err, "gaussian_process: "+op)
	}

	// Check matrix dimensions
	n, m := K.Dims()
	yRows, yCols := y.Dims()
	if n != m || n != nSamples || yRows != nSamples || yCols != 1 {
		err := fmt.Errorf("dimension mismatch in solveWithSVD: K(%d,%d), y(%d,%d), nSamples=%d",
			n, m, yRows, yCols, nSamples)
		return nil, optimization.WrapError(err, "gaussian_process: "+op)
	}

	// Log the condition number before SVD
	var svd mat.SVD
	ok := svd.Factorize(K, mat.SVDFull)
	if !ok {
		err := errors.New("SVD factorization failed")
		return nil, optimization.WrapError(err, "gaussian_process: "+op)
	}

	// Get singular values
	s := svd.Values(nil)
	if len(s) == 0 {
		err := errors.New("SVD returned no singular values")
		return nil, optimization.WrapError(err, "gaussian_process: "+op)
	}

	// Compute condition number
	maxS := s[0]
	minS := s[len(s)-1]
	cond := maxS / math.Max(minS, 1e-16) // Avoid division by zero

	gp.logger.Info("Using SVD solver",
		zap.Float64("condition_number", cond),
		zap.Float64("max_singular_value", maxS),
		zap.Float64("min_singular_value", minS),
	)

	// Compute U^T * y
	var U, V mat.Dense
	svd.UTo(&U)
	svd.VTo(&V)

	// Compute U^T * y
	var UTy mat.Dense
	UTy.Mul(U.T(), y)

	// Compute S^+ * (U^T * y)
	sPlusUTyData := make([]float64, nSamples)
	rank := 0
	threshold := math.Max(float64(nSamples), 1.0) * s[0] * 1e-15

	for i := 0; i < nSamples; i++ {
		if i < len(s) && s[i] > threshold {
			sPlusUTyData[i] = UTy.At(i, 0) / s[i]
			rank++
		} else {
			sPlusUTyData[i] = 0
		}
	}

	// Convert to diagonal matrix
	sPlusUTy := mat.NewDiagDense(nSamples, sPlusUTyData)

	// Compute V * S^+ * U^T * y
	var tmp mat.Dense
	tmp.Mul(&V, sPlusUTy)

	// Convert result to VecDense
	result := mat.NewVecDense(nSamples, nil)
	for i := 0; i < nSamples; i++ {
		result.SetVec(i, tmp.At(i, 0))
	}

	// Log the residual if needed
	if gp.logger.Level() <= zap.DebugLevel {
		var r mat.Dense
		r.Mul(K, result)
		r.Sub(&r, y)
		residual := mat.Norm(&r, 2)

		gp.logger.Debug("Solved system with SVD",
			zap.Float64("relative_residual", residual/mat.Norm(y, 2)),
			zap.Int("effective_rank", rank),
		)
	}

	if rank == 0 {
		err := errors.New("matrix is effectively rank zero after thresholding")
		return nil, optimization.WrapError(err, "gaussian_process: "+op)
	}

	return result, nil
}

// zeroMean is the default mean function that always returns zero
func zeroMean(x []float64) float64 {
	return 0.0
}
