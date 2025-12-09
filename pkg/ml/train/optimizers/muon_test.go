package optimizers

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gopjrt/dtypes"
)

func TestNewtonSchulzOrthogonalizeSquare(t *testing.T) {
	graphtest.RunTestGraphFn(t, "NewtonSchulz-Square", func(g *Graph) (inputs, outputs []*Node) {
		input := Const(g, [][]float32{
			{1.0, 2.0, 0.5},
			{0.3, 1.5, 0.8},
			{0.7, 0.2, 2.0},
		})
		inputs = []*Node{input}

		U := newtonSchulzOrthogonalize(input, 10)
		UtU := MatMul(transposeMatrix(U), U)
		outputs = []*Node{UtU}
		return
	}, []any{
		[][]float32{
			{1.0, 0.0, 0.0},
			{0.0, 1.0, 0.0},
			{0.0, 0.0, 1.0},
		},
	}, 1e-4)
}

func TestNewtonSchulzOrthogonalizeTall(t *testing.T) {
	graphtest.RunTestGraphFn(t, "NewtonSchulz-Tall", func(g *Graph) (inputs, outputs []*Node) {
		input := Const(g, [][]float32{
			{1.0, 2.0},
			{0.3, 1.5},
			{0.7, 0.2},
			{1.2, 0.9},
		})
		inputs = []*Node{input}

		U := newtonSchulzOrthogonalize(input, 10)
		UtU := MatMul(transposeMatrix(U), U)
		outputs = []*Node{UtU}
		return
	}, []any{
		[][]float32{
			{1.0, 0.0},
			{0.0, 1.0},
		},
	}, 1e-4)
}

func TestNewtonSchulzOrthogonalizeWide(t *testing.T) {
	graphtest.RunTestGraphFn(t, "NewtonSchulz-Wide", func(g *Graph) (inputs, outputs []*Node) {
		input := Const(g, [][]float32{
			{1.0, 2.0, 0.5, 1.1},
			{0.3, 1.5, 0.8, 0.6},
		})
		inputs = []*Node{input}

		U := newtonSchulzOrthogonalize(input, 10)
		UUt := MatMul(U, transposeMatrix(U))
		outputs = []*Node{UUt}
		return
	}, []any{
		[][]float32{
			{1.0, 0.0},
			{0.0, 1.0},
		},
	}, 1e-4)
}

func TestMuonConfig(t *testing.T) {
	cfg := Muon().
		LearningRate(0.01).
		Beta(0.9).
		NSIterations(7).
		Scope("test_muon")

	if cfg.learningRate != 0.01 {
		t.Errorf("expected lr 0.01, got %f", cfg.learningRate)
	}
	if cfg.beta != 0.9 {
		t.Errorf("expected beta 0.9, got %f", cfg.beta)
	}
	if cfg.nsIterations != 7 {
		t.Errorf("expected 7 iterations, got %d", cfg.nsIterations)
	}
	if cfg.scopeName != "test_muon" {
		t.Errorf("expected scope test_muon, got %s", cfg.scopeName)
	}
}

func TestIdentityMatrix(t *testing.T) {
	graphtest.RunTestGraphFn(t, "IdentityMatrix", func(g *Graph) (inputs, outputs []*Node) {
		I3 := identityMatrix(g, dtypes.Float32, 3)
		I2 := identityMatrix(g, dtypes.Float32, 2)
		outputs = []*Node{I3, I2}
		return
	}, []any{
		[][]float32{
			{1, 0, 0},
			{0, 1, 0},
			{0, 0, 1},
		},
		[][]float32{
			{1, 0},
			{0, 1},
		},
	}, 1e-6)
}
