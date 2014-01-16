package word2vec

import "math"

type Vector []float32

// Normalize this vector.
func (v Vector) Normalize() {
	w := float32(math.Sqrt(float64(dot(v, v))))
	for i := range v {
		v[i] = v[i] / w
	}
}

func add(x, y Vector, m float32) {
	for i, v := range y {
		x[i] += v * m
	}
}

func dot(x, y Vector) float32 {
	var w float32
	for i, v := range x {
		w += v * y[i]
	}
	return w
}
