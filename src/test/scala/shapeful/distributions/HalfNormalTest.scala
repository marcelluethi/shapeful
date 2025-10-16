package shapeful.distributions

import munit.FunSuite
import shapeful.*
import shapeful.tensor.Shape.*
import math.{Pi, abs, exp, log, sqrt}

class HalfNormalTests extends FunSuite:

  // Type aliases for better readability
  type Batch = "batch"
  type Feature = "feature"

  // Test tolerance for floating point comparisons
  val tolerance = 1e-5f

  def assertApproxEqual(actual: Float, expected: Float, tolerance: Float = tolerance): Unit =
    if abs(actual - expected) > tolerance then fail(s"Expected $expected, but got $actual (tolerance: $tolerance)")

  override def beforeAll(): Unit =
    super.beforeAll()

  test("logpdf computation for standard half-normal at x=0") {
    val sigma = Tensor0(1.0f)
    val halfNormal = HalfNormal(sigma)

    val x = Tensor0(0.0f)
    val logpdf = halfNormal.logpdf(x)

    // For half-normal at x=0 with σ=1: log(2) - 0.5 * log(2π) - 0
    val expected = log(2).toFloat - 0.5f * log(2 * Pi).toFloat
    assertApproxEqual(logpdf.toFloat, expected)
  }

  test("logpdf computation for standard half-normal at x=1") {
    val sigma = Tensor0(1.0f)
    val halfNormal = HalfNormal(sigma)

    val x = Tensor0(1.0f)
    val logpdf = halfNormal.logpdf(x)

    // For half-normal at x=1 with σ=1: log(2) - 0.5 * log(2π) - 0.5 * 1^2
    val expected = log(2).toFloat - 0.5f * log(2 * Pi).toFloat - 0.5f
    assertApproxEqual(logpdf.toFloat, expected)
  }

  test("logpdf computation for non-standard half-normal") {
    val sigma = Tensor0(2.0f)
    val halfNormal = HalfNormal(sigma)

    val x = Tensor0(1.0f)
    val logpdf = halfNormal.logpdf(x)

    // For half-normal at x=1 with σ=2: log(2) - 0.5 * log(2π) - log(2) - 0.5 * (1/2)^2
    val expected = log(2).toFloat - 0.5f * log(2 * Pi).toFloat - log(2).toFloat - 0.5f * 0.25f
    assertApproxEqual(logpdf.toFloat, expected)
  }

  test("sample generation produces correct shape") {
    val shape = Shape2[Batch, Feature](2, 3)
    val sigma = Tensor.ones(shape)
    val halfNormal = HalfNormal(sigma)
    val key = shapeful.random.Random.Key(42)

    val sample = halfNormal.sample(key)

    assertEquals(sample.shape, shape)
    assertEquals(sample.dtype, DType.Float32)
  }

  test("sample generation with scalar sigma") {
    val sigma = Tensor0(1.0f)
    val halfNormal = HalfNormal(sigma)
    val key = shapeful.random.Random.Key(123)

    val sample = halfNormal.sample(key)

    // Should be a scalar tensor
    assertEquals(sample.shape.dims.length, 0)
    assertEquals(sample.dtype, DType.Float32)
  }

  test("samples are non-negative") {
    val sigma = Tensor0(1.0f)
    val halfNormal = HalfNormal(sigma)
    val mainKey = shapeful.random.Random.Key(456)

    // Generate multiple samples to test non-negativity
    val keys = mainKey.split(10)
    for i <- 1 to 10 do
      val sample = halfNormal.sample(keys(i - 1))
      assert(sample.toFloat >= 0.0f, s"Sample ${sample.toFloat} should be non-negative")
  }

  test("logpdf increases with smaller sigma") {
    val x = Tensor0(1.0f)

    val halfNormal1 = HalfNormal(Tensor0(0.5f))
    val halfNormal2 = HalfNormal(Tensor0(2.0f))

    val logpdf1 = halfNormal1.logpdf(x)
    val logpdf2 = halfNormal2.logpdf(x)

    // Smaller sigma should give larger log probability density at the same point
    assert(logpdf2.toFloat > logpdf1.toFloat)
  }

  test("logpdf computation for tensor inputs") {
    val shape = Shape1[Batch](3)
    val sigma = Tensor(shape, Seq(1.0f, 1.0f, 1.0f), DType.Float32)
    val halfNormal = HalfNormal(sigma)

    val x = Tensor(shape, Seq(0.0f, 1.0f, 2.0f), DType.Float32)
    val logpdfElements = halfNormal.logpdfElements(x)

    // Check that we get a tensor of the same shape
    assertEquals(logpdfElements.shape, shape)
    assertEquals(logpdfElements.dtype, DType.Float32)
  }
