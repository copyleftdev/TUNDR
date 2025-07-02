package bayesian

import "gonum.org/v1/gonum/mat"

// MatrixPool provides a pool of reusable matrices to reduce allocations
type MatrixPool struct {
	symPools  []*mat.SymDense
	densePools []*mat.Dense
	vecPools  []*mat.VecDense
}

// NewMatrixPool creates a new MatrixPool
func NewMatrixPool() *MatrixPool {
	return &MatrixPool{
		symPools:  make([]*mat.SymDense, 0, 10),
		densePools: make([]*mat.Dense, 0, 10),
		vecPools:  make([]*mat.VecDense, 0, 10),
	}
}

// GetSymDense returns a symmetric matrix from the pool or creates a new one
func (p *MatrixPool) GetSymDense(n int) *mat.SymDense {
	if len(p.symPools) > 0 {
		m := p.symPools[len(p.symPools)-1]
		p.symPools = p.symPools[:len(p.symPools)-1]
		return m
	}
	return mat.NewSymDense(n, nil)
}

// PutSymDense returns a symmetric matrix to the pool
func (p *MatrixPool) PutSymDense(m *mat.SymDense) {
	p.symPools = append(p.symPools, m)
}

// GetDense returns a dense matrix from the pool or creates a new one
func (p *MatrixPool) GetDense(r, c int) *mat.Dense {
	if len(p.densePools) > 0 {
		m := p.densePools[len(p.densePools)-1]
		p.densePools = p.densePools[:len(p.densePools)-1]
		return m
	}
	return mat.NewDense(r, c, nil)
}

// PutDense returns a dense matrix to the pool
func (p *MatrixPool) PutDense(m *mat.Dense) {
	p.densePools = append(p.densePools, m)
}

// GetVecDense returns a vector from the pool or creates a new one
func (p *MatrixPool) GetVecDense(n int) *mat.VecDense {
	if len(p.vecPools) > 0 {
		v := p.vecPools[len(p.vecPools)-1]
		p.vecPools = p.vecPools[:len(p.vecPools)-1]
		return v
	}
	return mat.NewVecDense(n, nil)
}

// PutVecDense returns a vector to the pool
func (p *MatrixPool) PutVecDense(v *mat.VecDense) {
	p.vecPools = append(p.vecPools, v)
}
