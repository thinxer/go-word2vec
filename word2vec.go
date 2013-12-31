package word2vec

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

type Vector []float32

type Model struct {
	Layer1Size int
	Vocab      map[string]int

	data []float32
}

type Pair struct {
	Word string
	Sim  float32
}

func (v Vector) Unit() {
	w := float32(math.Sqrt(float64(dot(v, v))))
	for i := range v {
		v[i] = v[i] / w
	}
}

func LoadModel(filename string) (*Model, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	reader := bufio.NewReader(file)
	var vocabSize, layer1Size int
	fmt.Fscanln(reader, &vocabSize, &layer1Size)
	var word string
	model := &Model{
		Layer1Size: layer1Size,
		Vocab:      make(map[string]int),
		data:       make([]float32, layer1Size*vocabSize),
	}
	for i := 0; i < vocabSize; i++ {
		var vector = model.Vector(i)
		bytes, err := reader.ReadBytes(' ')
		if err != nil {
			return nil, err
		}
		word = string(bytes[:len(bytes)-1])
		err = binary.Read(reader, binary.LittleEndian, vector)
		if err != nil {
			return nil, err
		}
		vector.Unit()
		reader.ReadByte()

		model.Vocab[word] = i
	}
	return model, nil
}

func (m *Model) Vector(i int) Vector {
	return Vector(m.data[m.Layer1Size*i : m.Layer1Size*(i+1)])
}

func (m *Model) Similarity(x, y string) (float32, error) {
	id1, ok := m.Vocab[x]
	if !ok {
		return 0, fmt.Errorf("Word not found: %s", x)
	}
	id2, ok := m.Vocab[y]
	if !ok {
		return 0, fmt.Errorf("Word not found: %s", y)
	}
	return dot(m.Vector(id1), m.Vector(id2)), nil
}

func (m *Model) MostSimilar(word string, n int) ([]Pair, error) {
	wordId, ok := m.Vocab[word]
	if !ok {
		return nil, fmt.Errorf("Word not found: %s", word)
	}
	vec := m.Vector(wordId)
	r := make([]Pair, n)
	for w, i := range m.Vocab {
		if i == wordId {
			continue
		}
		sim := dot(m.Vector(i), vec)
		this := Pair{w, sim}
		for j := 0; j < n; j++ {
			if this.Sim > r[j].Sim {
				this, r[j] = r[j], this
			}
		}
	}
	return r, nil
}

func dot(x, y Vector) float32 {
	var w float32
	for i, v := range x {
		w += v * y[i]
	}
	return w
}
