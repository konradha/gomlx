package optimizers

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

func TestNewtonSchulzSquare(t *testing.T) {
	graphtest.RunTestGraphFn(t, "NewtonSchulz-Square", func(g *Graph) (inputs, outputs []*Node) {
		input := Const(g, [][]float32{
			{1.0, 2.0, 0.5},
			{0.3, 1.5, 0.8},
			{0.7, 0.2, 2.0},
		})
		U := newtonSchulzOrthogonalize(input, 5)
		UUt := MatMul(U, transposeMatrix(U))
		trace := ReduceAllSum(Mul(UUt, UUt))
		outputs = []*Node{trace}
		return
	}, []any{
		float32(3.0),
	}, 1.5)
}

func TestNewtonSchulzTall(t *testing.T) {
	graphtest.RunTestGraphFn(t, "NewtonSchulz-Tall", func(g *Graph) (inputs, outputs []*Node) {
		input := Const(g, [][]float32{
			{1.0, 2.0},
			{0.3, 1.5},
			{0.7, 0.2},
			{1.2, 0.9},
		})
		U := newtonSchulzOrthogonalize(input, 5)
		UtU := MatMul(transposeMatrix(U), U)
		trace := ReduceAllSum(Mul(UtU, UtU))
		outputs = []*Node{trace}
		return
	}, []any{
		float32(2.0),
	}, 1.0)
}

func TestNewtonSchulzWide(t *testing.T) {
	graphtest.RunTestGraphFn(t, "NewtonSchulz-Wide", func(g *Graph) (inputs, outputs []*Node) {
		input := Const(g, [][]float32{
			{1.0, 2.0, 0.5, 1.1},
			{0.3, 1.5, 0.8, 0.6},
		})
		U := newtonSchulzOrthogonalize(input, 5)
		UUt := MatMul(U, transposeMatrix(U))
		trace := ReduceAllSum(Mul(UUt, UUt))
		outputs = []*Node{trace}
		return
	}, []any{
		float32(2.0),
	}, 1.0)
}

func TestNewtonSchulz4DReshaped(t *testing.T) {
	graphtest.RunTestGraphFn(t, "NewtonSchulz-4D", func(g *Graph) (inputs, outputs []*Node) {
		input := Const(g, [][]float32{
			{1.0, 2.0, 0.5, 1.1, 0.3, 0.8},
			{0.3, 1.5, 0.8, 0.6, 1.2, 0.4},
			{0.7, 0.2, 2.0, 0.9, 0.5, 1.1},
			{1.2, 0.9, 0.4, 1.8, 0.7, 0.3},
		})
		U := newtonSchulzOrthogonalize(input, 5)
		UUt := MatMul(U, transposeMatrix(U))
		trace := ReduceAllSum(Mul(UUt, UUt))
		outputs = []*Node{trace}
		return
	}, []any{
		float32(4.0),
	}, 2.0)
}

func TestNewtonSchulzPreservesShape(t *testing.T) {
	graphtest.RunTestGraphFn(t, "NewtonSchulz-Shape", func(g *Graph) (inputs, outputs []*Node) {
		input := Const(g, [][]float32{
			{1.0, 2.0, 3.0},
			{4.0, 5.0, 6.0},
		})
		U := newtonSchulzOrthogonalize(input, 5)
		shape := U.Shape()
		rows := Const(g, float32(shape.Dim(0)))
		cols := Const(g, float32(shape.Dim(1)))
		outputs = []*Node{rows, cols}
		return
	}, []any{
		float32(2.0),
		float32(3.0),
	}, 0)
}

func TestNewtonSchulzNonZeroOutput(t *testing.T) {
	graphtest.RunTestGraphFn(t, "NewtonSchulz-NonZero", func(g *Graph) (inputs, outputs []*Node) {
		input := Const(g, [][]float32{
			{1.0, 2.0},
			{3.0, 4.0},
		})
		U := newtonSchulzOrthogonalize(input, 5)
		sumAbs := ReduceAllSum(Abs(U))
		outputs = []*Node{sumAbs}
		return
	}, []any{
		float32(2.0),
	}, 1.5)
}

func TestMuonReducesLoss(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()

	target := []float32{1, 0, -1, 0, 0.5, -0.5, 0.25, -0.25}

	computeLoss := func(ctx *context.Context, g *Graph) *Node {
		w := ctx.VariableWithShape("weights", shapes.Make(dtypes.Float32, 2, 2, 2, 1)).ValueGraph(g)
		flat := Reshape(w, 8)
		tgt := Const(g, target)
		diff := Sub(flat, tgt)
		return ReduceAllSum(Mul(diff, diff))
	}

	execLoss, err := context.NewExecAny(backend, ctx, computeLoss)
	if err != nil {
		t.Fatalf("failed to create execLoss: %v", err)
	}
	results, err := execLoss.Exec()
	if err != nil {
		t.Fatalf("execLoss.Exec failed: %v", err)
	}
	initialLoss := tensors.ToScalar[float32](results[0])

	opt := Muon().LearningRate(0.1).Momentum(0.6).Nesterov(true).NSIterations(5).Done()

	execTrain, err := context.NewExecAny(backend, ctx, func(ctx *context.Context, g *Graph) *Node {
		loss := computeLoss(ctx, g)
		opt.UpdateGraph(ctx, g, loss)
		return loss
	})
	if err != nil {
		t.Fatalf("failed to create execTrain: %v", err)
	}

	for range 50 {
		_, err = execTrain.Exec()
		if err != nil {
			t.Fatalf("execTrain.Exec failed: %v", err)
		}
	}

	results, err = execLoss.Exec()
	if err != nil {
		t.Fatalf("final execLoss.Exec failed: %v", err)
	}
	finalLoss := tensors.ToScalar[float32](results[0])

	if finalLoss >= initialLoss*0.5 {
		t.Errorf("Muon failed to significantly reduce loss: initial=%f, final=%f", initialLoss, finalLoss)
	}
}
