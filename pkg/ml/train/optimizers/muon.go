package optimizers

import (
	"fmt"

	. "github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
	"github.com/gomlx/gopjrt/dtypes"
)

const (
	MuonDefaultLearningRate = 0.02
	MuonDefaultScope        = "MuonOptimizer"
)

type MuonConfig struct {
	scopeName    string
	dtype        dtypes.DType
	learningRate float64
	momentum     float64
	nesterov     bool
	nsIterations int
}

func Muon() *MuonConfig {
	return &MuonConfig{
		scopeName:    MuonDefaultScope,
		learningRate: -1,
		momentum:     0.6,
		nesterov:     true,
		nsIterations: 5,
		dtype:        dtypes.InvalidDType,
	}
}

func (c *MuonConfig) FromContext(ctx *context.Context) *MuonConfig {
	c.momentum = context.GetParamOr(ctx, ParamMuonMomentum, c.momentum)
	c.nsIterations = context.GetParamOr(ctx, ParamMuonNSIterations, c.nsIterations)
	return c
}

func (c *MuonConfig) Scope(name string) *MuonConfig {
	c.scopeName = name
	return c
}

func (c *MuonConfig) DType(dtype dtypes.DType) *MuonConfig {
	c.dtype = dtype
	return c
}

func (c *MuonConfig) LearningRate(value float64) *MuonConfig {
	c.learningRate = value
	return c
}

func (c *MuonConfig) Momentum(m float64) *MuonConfig {
	c.momentum = m
	return c
}

func (c *MuonConfig) Nesterov(n bool) *MuonConfig {
	c.nesterov = n
	return c
}

func (c *MuonConfig) NSIterations(n int) *MuonConfig {
	c.nsIterations = n
	return c
}

func (c *MuonConfig) Done() Interface {
	return &muon{config: c}
}

type muon struct {
	config *MuonConfig
}

func (o *muon) UpdateGraph(ctx *context.Context, g *Graph, loss *Node) {
	if !loss.Shape().IsScalar() {
		Panicf("optimizer requires a scalar loss to optimize, got loss.shape=%s instead", loss.Shape())
	}
	grads := ctx.BuildTrainableVariablesGradientsGraph(loss)
	o.UpdateGraphWithGradients(ctx, grads, loss.DType())
}

func (o *muon) UpdateGraphWithGradients(ctx *context.Context, grads []*Node, lossDType dtypes.DType) {
	if len(grads) == 0 {
		Panicf("no gradients provided, are there any trainable variables?")
	}
	g := grads[0].Graph()

	dtype := o.config.dtype
	if dtype == dtypes.InvalidDType {
		dtype = lossDType
	}

	lrValue := o.config.learningRate
	if lrValue < 0 {
		lrValue = context.GetParamOr(ctx, ParamLearningRate, MuonDefaultLearningRate)
	}
	lrVar := LearningRateVar(ctx, dtype, lrValue)
	learningRate := lrVar.ValueGraph(g)

	_ = IncrementGlobalStepGraph(ctx, g, dtype)

	momentum := Const(g, shapes.CastAsDType(o.config.momentum, dtype))

	numTrainable := len(grads)
	varIdx := 0
	for v := range ctx.IterVariables() {
		if v.Trainable && v.InUseByGraph(g) {
			if varIdx < numTrainable {
				if v.Shape().Rank() >= 2 {
					o.applyMuonGraph(ctx, g, v, dtype, grads[varIdx], learningRate, momentum)
				} else {
					o.applyFallbackGraph(ctx, g, v, dtype, grads[varIdx], learningRate, momentum)
				}
			}
			varIdx++
		}
	}
	if varIdx != numTrainable {
		Panicf("gradient count mismatch: got %d gradients but found %d trainable variables", numTrainable, varIdx)
	}
}

func transposeMatrix(X *Node) *Node {
	return Transpose(X, 1, 0)
}

// newtonSchulzOrthogonalize computes the orthogonal polar factor via Newton-Schulz approx
// For G=USV.T ~> U V.T
// https://arxiv.org/abs/2502.16982
func newtonSchulzOrthogonalize(X *Node, iterations int) *Node {
	g := X.Graph()
	dtype := X.DType()

	a := Const(g, shapes.CastAsDType(3.4445, dtype))
	b := Const(g, shapes.CastAsDType(-4.7750, dtype))
	c := Const(g, shapes.CastAsDType(2.0315, dtype))

	norm := Sqrt(ReduceAllSum(Square(X)))
	eps := Const(g, shapes.CastAsDType(1e-7, dtype))
	X = Div(X, Add(norm, eps))

	shape := X.Shape()
	m, n := shape.Dim(0), shape.Dim(1)
	transposed := m > n

	if transposed {
		X = transposeMatrix(X)
	}

	for range iterations {
		A := MatMul(X, transposeMatrix(X))
		B := Add(Mul(b, A), Mul(c, MatMul(A, A)))
		X = Add(Mul(a, X), MatMul(B, X))
	}

	if transposed {
		X = transposeMatrix(X)
	}

	return X
}

func (o *muon) applyMuonGraph(ctx *context.Context, g *Graph, v *context.Variable, dtype dtypes.DType,
	grad *Node, learningRate, momentum *Node) {

	if grad.DType() != dtype {
		grad = ConvertDType(grad, dtype)
	}
	TraceNaNInGradients(ctx, v, grad)
	grad = ClipNaNsInGradients(ctx, grad)

	mVar := o.getMomentumVariable(ctx, v, dtype)
	buf := mVar.ValueGraph(g)

	newBuf := Add(Mul(momentum, buf), grad)
	mVar.SetValueGraph(newBuf)

	var update *Node
	if o.config.nesterov {
		update = Add(grad, Mul(momentum, newBuf))
	} else {
		update = newBuf
	}

	value := v.ValueGraph(g)
	if value.DType() != dtype {
		value = ConvertDType(value, dtype)
	}

	numElements := Const(g, shapes.CastAsDType(float64(v.Shape().Size()), dtype))
	valueNorm := Sqrt(ReduceAllSum(Square(value)))
	eps := Const(g, shapes.CastAsDType(1e-12, dtype))
	value = Mul(value, Div(Sqrt(numElements), Add(valueNorm, eps)))

	originalShape := update.Shape().Clone()
	var update2D *Node
	if originalShape.Rank() == 2 {
		update2D = update
	} else {
		rows := originalShape.Dim(0)
		cols := originalShape.Size() / rows
		update2D = Reshape(update, rows, cols)
	}

	whitened := newtonSchulzOrthogonalize(update2D, o.config.nsIterations)

	if originalShape.Rank() != 2 {
		whitened = Reshape(whitened, originalShape.Dimensions...)
	}

	step := Mul(learningRate, whitened)

	clipByValue := context.GetParamOr(ctx, ParamClipStepByValue, 0.0)
	if clipByValue > 0 {
		step = ClipScalar(step, -clipByValue, clipByValue)
	}

	updated := Sub(value, step)
	updated = ClipNaNsInUpdates(ctx, value, updated)

	if v.Shape().DType != dtype {
		updated = ConvertDType(updated, v.Shape().DType)
	}
	v.SetValueGraph(updated)
}

func (o *muon) applyFallbackGraph(ctx *context.Context, g *Graph, v *context.Variable, dtype dtypes.DType,
	grad *Node, learningRate, momentum *Node) {

	if grad.DType() != dtype {
		grad = ConvertDType(grad, dtype)
	}
	TraceNaNInGradients(ctx, v, grad)
	grad = ClipNaNsInGradients(ctx, grad)

	mVar := o.getMomentumVariable(ctx, v, dtype)
	buf := mVar.ValueGraph(g)

	newBuf := Add(Mul(momentum, buf), grad)
	mVar.SetValueGraph(newBuf)

	var update *Node
	if o.config.nesterov {
		update = Add(grad, Mul(momentum, newBuf))
	} else {
		update = newBuf
	}

	value := v.ValueGraph(g)
	if value.DType() != dtype {
		value = ConvertDType(value, dtype)
	}

	step := Mul(learningRate, update)

	clipByValue := context.GetParamOr(ctx, ParamClipStepByValue, 0.0)
	if clipByValue > 0 {
		step = ClipScalar(step, -clipByValue, clipByValue)
	}

	updated := Sub(value, step)
	updated = ClipNaNsInUpdates(ctx, value, updated)

	if v.Shape().DType != dtype {
		updated = ConvertDType(updated, v.Shape().DType)
	}
	v.SetValueGraph(updated)
}

func (o *muon) getMomentumVariable(ctx *context.Context, trainable *context.Variable, dtype dtypes.DType) *context.Variable {
	originalScope := trainable.Scope()
	originalName := trainable.Name()
	scopePath := fmt.Sprintf("%s%s%s", context.ScopeSeparator, o.config.scopeName, originalScope)

	shape := trainable.Shape().Clone()
	shape.DType = dtype

	ctx = ctx.Checked(false)

	return ctx.InAbsPath(scopePath).
		WithInitializer(initializers.Zero).
		VariableWithShape(fmt.Sprintf("%s_momentum", originalName), shape).
		SetTrainable(false)
}

func (o *muon) Clear(ctx *context.Context) error {
	return ctx.In(o.config.scopeName).DeleteVariablesInScope()
}
