package shapeful.autodiff

import scala.language.experimental.namedTypeArguments

import munit.FunSuite

import shapeful.*
import shapeful.jax.Jax
import shapeful.tensor.DType
import me.shadaj.scalapy.py.SeqConverters

class AutodiffTests extends FunSuite:

  // Test fixtures - define common labels for reuse
  type Feature = "feature"
  type Weight = "weight"
  type Bias = "bias"
  type Input = "input"
  type Output = "output"

  override def beforeAll(): Unit =
    // Initialize Python/JAX environment if needed
    super.beforeAll()

  test("grad of simple quadratic function") {
    // f(x) = x^2, f'(x) = 2x
    val f = (x: Tensor0) => x * x
    val gradF = Autodiff.grad(f)

    val x = Tensor0(3.0f)
    val gradient = gradF(x)

    // Expected: 2 * 3 = 6
    val expected = Tensor0(6.0f)
    assert(gradient.approxEquals(expected, tolerance = 1e-5f), s"Expected ${expected.toFloat}, got ${gradient.toFloat}")
  }

  test("grad of cubic function") {
    // f(x) = x^3, f'(x) = 3x^2
    val f = (x: Tensor0) => x * x * x
    val gradF = Autodiff.grad(f)

    val x = Tensor0(2.0f)
    val gradient = gradF(x)

    // Expected: 3 * 2^2 = 12
    val expected = Tensor0(12.0f)
    assert(gradient.approxEquals(expected, tolerance = 1e-5f), s"Expected ${expected.toFloat}, got ${gradient.toFloat}")
  }

  test("grad of linear function") {
    // f(x) = 3x + 2, f'(x) = 3
    val f = (x: Tensor0) => x * Tensor0(3.0f) + Tensor0(2.0f)
    val gradF = Autodiff.grad(f)

    val x = Tensor0(5.0f) // Value doesn't matter for linear function
    val gradient = gradF(x)

    // Expected: 3
    val expected = Tensor0(3.0f)
    assert(gradient.approxEquals(expected, tolerance = 1e-5f), s"Expected ${expected.toFloat}, got ${gradient.toFloat}")
  }

  test("grad of exponential function") {
    // f(x) = exp(x), f'(x) = exp(x)
    val f = (x: Tensor0) => x.exp
    val gradF = Autodiff.grad(f)

    val x = Tensor0(1.0f)
    val gradient = gradF(x)
    val functionValue = f(x)

    // For exp(x), derivative equals function value
    assert(
      gradient.approxEquals(functionValue, tolerance = 1e-5f),
      s"Expected ${functionValue.toFloat}, got ${gradient.toFloat}"
    )
  }

  test("value_and_grad computes both value and gradient") {
    // f(x) = x^3 + 2x, f'(x) = 3x^2 + 2
    val f = (x: Tensor0) => x * x * x + Tensor0(2.0f) * x
    val valueAndGradF = Autodiff.valueAndGrad(f)

    val x = Tensor0(2.0f)
    val (value, gradient) = valueAndGradF(x)

    // Expected value: f(2) = 8 + 4 = 12
    val expectedValue = Tensor0(12.0f)
    assert(
      value.approxEquals(expectedValue, tolerance = 1e-5f),
      s"Value: expected ${expectedValue.toFloat}, got ${value.toFloat}"
    )

    // Expected gradient: f'(2) = 3*4 + 2 = 14
    val expectedGradient = Tensor0(14.0f)
    assert(
      gradient.approxEquals(expectedGradient, tolerance = 1e-5f),
      s"Gradient: expected ${expectedGradient.toFloat}, got ${gradient.toFloat}"
    )

    // Verify it matches separate computations
    val separateValue = f(x)
    val separateGradient = Autodiff.grad(f)(x)

    assert(
      value.approxEquals(separateValue, tolerance = 1e-6f),
      "value_and_grad value should match separate function evaluation"
    )
    assert(
      gradient.approxEquals(separateGradient, tolerance = 1e-6f),
      "value_and_grad gradient should match separate gradient computation"
    )
  }

  test("jacFwd computes Jacobian using forward-mode differentiation") {
    // Vector function using available operations: f(x) = [sum(x^2), mean(x), norm(x)]
    val f = (x: Tensor1[Feature]) =>
      val squared = x * x
      val sumSquared = squared.sum // Sum of squares
      val meanVal = x.mean // Mean value
      val normVal = x.norm // L2 norm
      Tensor.stack[NewAxis = Output](sumSquared, Seq(meanVal, normVal))

    val jacF = Autodiff.jacFwd[Feature, Output](f)

    val x = Tensor1[Feature](Seq(2.0f, 3.0f))
    val jacobian: Tensor2[Output, Feature] = jacF(x)

    // Expected Jacobian:
    // f(x) = [x0^2 + x1^2, (x0 + x1)/2, sqrt(x0^2 + x1^2)]
    // For x = [2, 3]:
    // df1/dx0 = 2*x0 = 4, df1/dx1 = 2*x1 = 6
    // df2/dx0 = 0.5, df2/dx1 = 0.5
    // df3/dx0 = x0 / sqrt(x0^2 + x1^2) = 2/√13
    // df3/dx1 = x1 / sqrt(x0^2 + x1^2) = 3/√13
    val sqrt13 = math.sqrt(13.0).toFloat
    val expected = Tensor2[Output, Feature](
      Seq(
        Seq(4.0f, 6.0f),
        Seq(0.5f, 0.5f),
        Seq(2.0f / sqrt13, 3.0f / sqrt13)
      )
    )

    // Verify the computation runs and produces correct shape
    assert(jacobian.shape.dims.length == 2, "Jacobian should be a 2D tensor")
    assert(jacobian.shape.dims(0) == 3, "Output dimension should be 3")
    assert(jacobian.shape.dims(1) == 2, "Input dimension should be 2")
    // Check actual values
    assert(
      jacobian.approxEquals(expected, tolerance = 1e-5f),
      s"Jacobian values incorrect.\nExpected: ${expected.inspect}\nActual: ${jacobian.inspect}"
    )
  }

  test("jacRev computes Jacobian using reverse-mode differentiation") {
    // Simple vector function using available operations: f(x) = [2*sum(x), mean(x)]
    val f = (x: Tensor1[Feature]) =>
      val scaledSum = x.sum * Tensor0(2.0f) // 2 * sum
      val meanVal = x.mean // Mean value
      Tensor1[Output](Seq(scaledSum.toFloat, meanVal.toFloat))

    val jacF = Autodiff.jacRev[Feature, Output](f)

    val x = Tensor1[Feature](Seq(3.0f, 4.0f))
    val jacobian: Tensor2[Output, Feature] = jacF(x)

    // Verify the computation runs and produces a 2D tensor
    assert(jacobian.shape.dims.length == 2, "Jacobian should be a 2D tensor")
    assert(jacobian.shape.dims(0) == 2, "Output dimension should be 2")
    assert(jacobian.shape.dims(1) == 2, "Input dimension should be 2")

    // Test that jacFwd and jacRev give the same result for this function
    val jacFwdF = Autodiff.jacFwd[Feature, Output](f)
    val jacobianFwd: Tensor2[Output, Feature] = jacFwdF(x)

    assert(
      jacobian.approxEquals(jacobianFwd, tolerance = 1e-5f),
      "jacRev and jacFwd should give the same result"
    )
  }

  test("grad of vector sum") {
    // f(v) = sum(v), gradient should be vector of ones
    val f = (v: Tensor1[Feature]) => v.sum
    val gradF = Autodiff.grad(f)

    val v = Tensor1[Feature](Seq(1.0f, 2.0f, 3.0f, 4.0f))
    val gradient = gradF(v)

    // Expected: [1, 1, 1, 1]
    val expected = Tensor1[Feature](Seq(1.0f, 1.0f, 1.0f, 1.0f))
    assert(gradient.approxEquals(expected, tolerance = 1e-5f), s"Expected ones vector, got ${gradient.toString}")
  }

  test("grad of vector norm squared") {
    // f(v) = ||v||^2 = sum(v^2), f'(v) = 2v
    val f = (v: Tensor1[Feature]) =>
      val squared = v * v
      squared.sum
    val gradF = Autodiff.grad(f)

    val v = Tensor1[Feature](Seq(1.0f, 2.0f, 3.0f))
    val gradient = gradF(v)

    // Expected: 2 * [1, 2, 3] = [2, 4, 6]
    val expected = Tensor1[Feature](Seq(2.0f, 4.0f, 6.0f))
    assert(gradient.approxEquals(expected, tolerance = 1e-5f), s"Expected 2*v, got ${gradient.toString}")
  }

  test("grad of simple linear regression loss") {
    // Linear model: y = w * x + b
    // Loss: (y - target)^2
    case class LinearModel(w: Tensor0, b: Tensor0)

    given ToPyTree[LinearModel] with
      def toPyTree(model: LinearModel): Jax.PyAny =
        me.shadaj.scalapy.py.Dynamic.global.tuple(Seq(model.w.jaxValue, model.b.jaxValue).toPythonProxy)
      def fromPyTree(p: Jax.PyAny): LinearModel =
        val seq = p.as[Seq[me.shadaj.scalapy.py.Dynamic]]
        LinearModel(
          w = new Tensor[EmptyTuple](Shape.empty, seq(0), DType.Float32),
          b = new Tensor[EmptyTuple](Shape.empty, seq(1), DType.Float32)
        )

    val target = Tensor0(5.0f)
    val x = Tensor0(2.0f)

    val lossF = (model: LinearModel) =>
      val prediction = model.w * x + model.b
      val diff = prediction - target
      diff * diff

    val gradF = Autodiff.grad(lossF)

    // Test at w=1, b=1: prediction = 1*2 + 1 = 3, diff = 3-5 = -2
    // Loss = (-2)^2 = 4
    // dL/dw = 2 * diff * x = 2 * (-2) * 2 = -8
    // dL/db = 2 * diff * 1 = 2 * (-2) = -4
    val model = LinearModel(w = Tensor0(1.0f), b = Tensor0(1.0f))
    val gradient = gradF(model)

    assert(gradient.w.approxEquals(Tensor0(-8.0f), tolerance = 1e-4f), s"Expected w grad -8, got ${gradient.w.toFloat}")
    assert(gradient.b.approxEquals(Tensor0(-4.0f), tolerance = 1e-4f), s"Expected b grad -4, got ${gradient.b.toFloat}")
  }

  test("grad with multiple scalar operations") {
    // f(x, y) = x^2 + y^2 + x*y
    case class TwoScalars(x: Tensor0, y: Tensor0)

    given ToPyTree[TwoScalars] with
      def toPyTree(params: TwoScalars): Jax.PyAny =
        me.shadaj.scalapy.py.Dynamic.global.tuple(Seq(params.x.jaxValue, params.y.jaxValue).toPythonProxy)
      def fromPyTree(p: Jax.PyAny): TwoScalars =
        val seq = p.as[Seq[me.shadaj.scalapy.py.Dynamic]]
        TwoScalars(
          x = new Tensor[EmptyTuple](Shape.empty, seq(0), DType.Float32),
          y = new Tensor[EmptyTuple](Shape.empty, seq(1), DType.Float32)
        )

    val f = (params: TwoScalars) => params.x * params.x + params.y * params.y + params.x * params.y

    val gradF = Autodiff.grad(f)

    val params = TwoScalars(x = Tensor0(2.0f), y = Tensor0(3.0f))
    val gradient = gradF(params)

    // df/dx = 2x + y = 2*2 + 3 = 7
    // df/dy = 2y + x = 2*3 + 2 = 8
    assert(gradient.x.approxEquals(Tensor0(7.0f), tolerance = 1e-5f), s"Expected x grad 7, got ${gradient.x.toFloat}")
    assert(gradient.y.approxEquals(Tensor0(8.0f), tolerance = 1e-5f), s"Expected y grad 8, got ${gradient.y.toFloat}")
  }

  test("grad is consistent with finite differences") {
    // Verify gradients match finite difference approximation
    val f = (x: Tensor0) => x * x * x + x * x
    val gradF = Autodiff.grad(f)

    val x = Tensor0(1.5f)
    val analyticalGrad = gradF(x)

    // Finite difference approximation
    val h = 1e-5f
    val xPlusH = Tensor0(x.toFloat + h)
    val xMinusH = Tensor0(x.toFloat - h)
    val numericalGrad = (f(xPlusH).toFloat - f(xMinusH).toFloat) / (2 * h)

    assert(
      math.abs(analyticalGrad.toFloat - numericalGrad) < 1e-2f,
      s"Analytical grad: ${analyticalGrad.toFloat}, Numerical grad: $numericalGrad"
    )
  }

  test("second derivative by applying grad twice") {
    // f(x) = x^4, f'(x) = 4x^3, f''(x) = 12x^2
    val f = (x: Tensor0) => x * x * x * x

    // First derivative
    val gradF = Autodiff.grad(f)

    // Second derivative - gradient of the gradient
    val grad2F = Autodiff.grad(gradF)

    val x = Tensor0(2.0f)

    // First derivative at x=2: f'(2) = 4 * 2^3 = 32
    val firstDerivative = gradF(x)
    val expectedFirst = Tensor0(32.0f)
    assert(
      firstDerivative.approxEquals(expectedFirst, tolerance = 1e-5f),
      s"First derivative: expected ${expectedFirst.toFloat}, got ${firstDerivative.toFloat}"
    )

    // Second derivative at x=2: f''(2) = 12 * 2^2 = 48
    val secondDerivative = grad2F(x)
    val expectedSecond = Tensor0(48.0f)
    assert(
      secondDerivative.approxEquals(expectedSecond, tolerance = 1e-5f),
      s"Second derivative: expected ${expectedSecond.toFloat}, got ${secondDerivative.toFloat}"
    )
  }

end AutodiffTests
