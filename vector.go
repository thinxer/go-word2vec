package word2vec

import "github.com/ziutek/blas"

type Vector []float32

// Normalize this vector.
func (v Vector) Normalize() {
	w := blas.Snrm2(len(v), v, 1)
	blas.Sscal(len(v), 1/w, v, 1)
}

func (y Vector) Add(alpha float32, x Vector) {
	blas.Saxpy(len(y), alpha, x, 1, y, 1)
}

func (y Vector) Dot(x Vector) float32 {
	return blas.Sdot(len(y), x, 1, y, 1)
}
