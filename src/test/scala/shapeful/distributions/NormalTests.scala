package shapeful.distributions

import scala.language.experimental.namedTypeArguments
import munit.FunSuite
import shapeful.*
import shapeful.tensor.Shape.*
import math.{Pi, abs, exp, log, sqrt}

class NormalTests extends FunSuite:

  // Type aliases for better readability
  type Batch = "batch"
  type Feature = "feature"

  // Test tolerance for floating point comparisons
  val tolerance = 1e-5f

  def assertApproxEqual(actual: Float, expected: Float, tolerance: Float = tolerance): Unit =
    if abs(actual - expected) > tolerance then fail(s"Expected $expected, but got $actual (tolerance: $tolerance)")

  override def beforeAll(): Unit =
    super.beforeAll()

  test("Normal distribution creation with scalar parameters") {
    val mu = Tensor0(0.0f)
    val sigma = Tensor0(1.0f)
    val normal = Normal(mu, sigma)

    // Test that the distribution is created without errors
    assert(normal != null)
  }

  test("Normal distribution creation with tensor parameters") {
    val shape = Shape1[Batch](3)
    val mu = Tensor(shape, Seq(0.0f, 1.0f, -1.0f), DType.Float32)
    val sigma = Tensor(shape, Seq(1.0f, 2.0f, 0.5f), DType.Float32)
    val normal = Normal(mu, sigma)

    assert(normal != null)
  }

  test("logpdf computation for standard normal (scalar)") {
    val mu = Tensor0(0.0f)
    val sigma = Tensor0(1.0f)
    val normal = Normal(mu, sigma)

    // Test at x = 0 (should be peak of standard normal)
    val x = Tensor0(0.0f)
    val logpdf = normal.logpdf(x)

    // For standard normal at x=0: log(1/sqrt(2π)) = -0.5 * log(2π)
    val expected = -0.5f * log(2 * Pi).toFloat
    assertApproxEqual(logpdf.toFloat, expected)
  }

  test("logpdf computation for standard normal at x=1") {
    val mu = Tensor0(0.0f)
    val sigma = Tensor0(1.0f)
    val normal = Normal(mu, sigma)

    val x = Tensor0(1.0f)
    val logpdf = normal.logpdf(x)

    // For standard normal at x=1: -0.5 * log(2π) - 0.5 * 1^2
    val expected = -0.5f * log(2 * Pi).toFloat - 0.5f
    assertApproxEqual(logpdf.toFloat, expected)
  }

  test("logpdf computation for non-standard normal") {
    val mu = Tensor0(2.0f)
    val sigma = Tensor0(3.0f)
    val normal = Normal(mu, sigma)

    val x = Tensor0(2.0f) // At the mean
    val logpdf = normal.logpdf(x)

    // At the mean: -0.5 * log(2π) - log(σ)
    val expected = -0.5f * log(2 * Pi).toFloat - log(3.0f).toFloat
    assertApproxEqual(logpdf.toFloat, expected)
  }

  test("logpdf computation for tensor inputs") {
    val shape = Shape1[Batch](3)
    val mu = Tensor(shape, Seq(0.0f, 1.0f, -1.0f), DType.Float32)
    val sigma = Tensor(shape, Seq(1.0f, 1.0f, 1.0f), DType.Float32)
    val normal = Normal(mu, sigma)

    val x = Tensor(shape, Seq(0.0f, 1.0f, -1.0f), DType.Float32) // All at their respective means
    val logpdf = normal.logpdf(x)

    // All should give the same value since they're all at their means with same sigma
    val expected = -0.5f * log(2 * Pi).toFloat
    val expectedTensor = Tensor(shape, Seq(expected, expected, expected), DType.Float32)

    // Use tensor equality for comparison
    assert(logpdf.approxEquals(expectedTensor, tolerance))
  }

  test("logpdf symmetry around mean") {
    val mu = Tensor0(5.0f)
    val sigma = Tensor0(2.0f)
    val normal = Normal(mu, sigma)

    val x1 = Tensor0(3.0f) // mu - 2
    val x2 = Tensor0(7.0f) // mu + 2

    val logpdf1 = normal.logpdf(x1)
    val logpdf2 = normal.logpdf(x2)

    // Should be equal due to symmetry
    assertApproxEqual(logpdf1.toFloat, logpdf2.toFloat)
  }

  test("sample generation produces correct shape") {
    val shape = Shape2[Batch, Feature](2, 3)
    val mu = Tensor.zeros(shape)
    val sigma = Tensor.ones(shape)
    val normal = Normal(mu, sigma)

    val sample = normal.sample()

    assertEquals(sample.shape, shape)
    assertEquals(sample.dtype, DType.Float32)
  }

  test("sample generation with scalar parameters") {
    val mu = Tensor0(0.0f)
    val sigma = Tensor0(1.0f)
    val normal = Normal(mu, sigma)

    val sample = normal.sample()

    // Should be a scalar tensor
    assertEquals(sample.shape.dims.length, 0)
    assertEquals(sample.dtype, DType.Float32)
  }

  test("logpdf increases sigma decreases probability density") {
    val mu = Tensor0(0.0f)
    val x = Tensor0(0.0f)

    val normal1 = Normal(mu, Tensor0(1.0f))
    val normal2 = Normal(mu, Tensor0(2.0f))

    val logpdf1 = normal1.logpdf(x)
    val logpdf2 = normal2.logpdf(x)

    // Larger sigma should give smaller log probability density at the same point
    assert(logpdf1.toFloat > logpdf2.toFloat)
  }

  test("logpdf numerical stability for small probabilities") {
    val mu = Tensor0(0.0f)
    val sigma = Tensor0(1.0f)
    val normal = Normal(mu, sigma)

    // Test far from mean (should give very small probability)
    val x = Tensor0(10.0f)
    val logpdf = normal.logpdf(x)

    // Should be a large negative number, but finite
    assert(logpdf.toFloat.isFinite)
    assert(logpdf.toFloat < -10.0f) // Should be very negative
  }

  test("different mu values give different logpdf") {
    val sigma = Tensor0(1.0f)
    val x = Tensor0(0.0f)

    val normal1 = Normal(Tensor0(0.0f), sigma)
    val normal2 = Normal(Tensor0(1.0f), sigma)

    val logpdf1 = normal1.logpdf(x)
    val logpdf2 = normal2.logpdf(x)

    // Different means should give different log probabilities for the same x
    assert(logpdf1.toFloat != logpdf2.toFloat)
    // x=0 should be more likely under N(0,1) than N(1,1)
    assert(logpdf1.toFloat > logpdf2.toFloat)
  }

  test("logpdf with 2D tensor inputs") {
    val shape = Shape2[Batch, Feature](2, 2)
    val mu = Tensor.zeros(shape)
    val sigma = Tensor.ones(shape)
    val normal = Normal(mu, sigma)

    val x = Tensor.zeros(shape) // All at mean
    val logpdf = normal.logpdf(x)

    assertEquals(logpdf.shape, shape)

    // All values should be the same since all inputs are at their respective means
    val expected = -0.5f * log(2 * Pi).toFloat
    val expectedTensor = Tensor(shape, Seq.fill(4)(expected), DType.Float32)

    // Use tensor equality for comparison
    assert(logpdf.approxEquals(expectedTensor, tolerance))
  }
