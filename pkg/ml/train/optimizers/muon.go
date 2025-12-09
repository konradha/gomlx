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
	beta         float64
	nsIterations int
	epsilon      float64
}

func Muon() *MuonConfig {
	return &MuonConfig{
		scopeName:    MuonDefaultScope,
		learningRate: -1,
		beta:         0.95,
		nsIterations: 5,
		epsilon:      1e-7,
		dtype:        dtypes.InvalidDType,
	}
}

func (c *MuonConfig) FromContext(ctx *context.Context) *MuonConfig {
	c.beta = context.GetParamOr(ctx, ParamMuonBeta, c.beta)
	c.nsIterations = context.GetParamOr(ctx, ParamMuonNSIterations, c.nsIterations)
	c.epsilon = context.GetParamOr(ctx, ParamMuonEpsilon, c.epsilon)
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

func (c *MuonConfig) Beta(beta float64) *MuonConfig {
	c.beta = beta
	return c
}

func (c *MuonConfig) NSIterations(n int) *MuonConfig {
	c.nsIterations = n
	return c
}

func (c *MuonConfig) Epsilon(eps float64) *MuonConfig {
	c.epsilon = eps
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

	beta := Const(g, shapes.CastAsDType(o.config.beta, dtype))
	onePlusBeta := Const(g, shapes.CastAsDType(1.0+o.config.beta, dtype))

	numTrainable := len(grads)
	varIdx := 0
	for v := range ctx.IterVariables() {
		if v.Trainable && v.InUseByGraph(g) {
			if varIdx < numTrainable {
				if v.Shape().Rank() == 2 {
					o.applyMuonGraph(ctx, g, v, dtype, grads[varIdx], learningRate, beta, onePlusBeta)
				} else {
					o.applyFallbackGraph(ctx, g, v, dtype, grads[varIdx], learningRate)
				}
			}
			varIdx++
		}
	}
	if varIdx != numTrainable {
		Panicf("gradient count mismatch: got %d gradients but found %d trainable variables", numTrainable, varIdx)
	}
}

func identityMatrix(g *Graph, dtype dtypes.DType, n int) *Node {
	indices := Iota(g, shapes.Make(dtypes.Int32, n), 0)
	return ConvertDType(Equal(
		ExpandDims(indices, 1),
		ExpandDims(indices, 0),
	), dtype)
}

func transposeMatrix(X *Node) *Node {
	return Transpose(X, 1, 0)
}

// newtonSchulzOrthogonalize computes the orthogonal polar factor via Newton-Schulz approx
// For G=USV.T ~> U V.T
// https://arxiv.org/abs/2502.16982
func newtonSchulzOrthogonalize(X *Node, iterations int) *Node {
	shape := X.Shape()
	m, n := shape.Dim(0), shape.Dim(1)
	g := X.Graph()
	dtype := X.DType()

	frobNorm := Sqrt(ReduceAllSum(Square(X)))
	scale := Const(g, shapes.CastAsDType(1.0/float64(max(m, n)), dtype))
	X = Div(X, Add(frobNorm, Const(g, shapes.CastAsDType(1e-12, dtype))))
	X = Mul(X, Sqrt(scale))

	tall := m >= n

	for range iterations {
		if tall {
			XtX := MatMul(transposeMatrix(X), X)
			I := identityMatrix(g, dtype, n)
			factor := Sub(MulScalar(I, 3.0), XtX)
			X = MulScalar(MatMul(X, factor), 0.5)
		} else {
			XXt := MatMul(X, transposeMatrix(X))
			I := identityMatrix(g, dtype, m)
			factor := Sub(MulScalar(I, 3.0), XXt)
			X = MulScalar(MatMul(factor, X), 0.5)
		}
	}

	return X
}

func (o *muon) applyMuonGraph(ctx *context.Context, g *Graph, v *context.Variable, dtype dtypes.DType,
	grad *Node, learningRate, beta, onePlusBeta *Node) {

	mVar := o.getMomentumVariable(ctx, v, dtype)
	mCurr := mVar.ValueGraph(g)

	if grad.DType() != dtype {
		grad = ConvertDType(grad, dtype)
	}
	TraceNaNInGradients(ctx, v, grad)
	grad = ClipNaNsInGradients(ctx, grad)

	mNew := Add(Mul(beta, mCurr), grad)

	nesterov := Sub(Mul(onePlusBeta, mNew), Mul(beta, mCurr))

	U := newtonSchulzOrthogonalize(nesterov, o.config.nsIterations)

	value := v.ValueGraph(g)
	if value.DType() != dtype {
		value = ConvertDType(value, dtype)
	}

	step := Mul(learningRate, U)

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

	mVar.SetValueGraph(mNew)
}

func (o *muon) applyFallbackGraph(ctx *context.Context, g *Graph, v *context.Variable, dtype dtypes.DType,
	grad *Node, learningRate *Node) {

	if grad.DType() != dtype {
		grad = ConvertDType(grad, dtype)
	}
	TraceNaNInGradients(ctx, v, grad)
	grad = ClipNaNsInGradients(ctx, grad)

	value := v.ValueGraph(g)
	if value.DType() != dtype {
		value = ConvertDType(value, dtype)
	}

	step := Mul(learningRate, grad)

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
	ctxMuon := ctx.In(o.config.scopeName)
	return ctxMuon.DeleteVariablesInScope()
}
