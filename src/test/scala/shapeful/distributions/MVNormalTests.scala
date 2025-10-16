package shapeful.distributions

import munit.FunSuite
import shapeful.*
import shapeful.tensor.Shape.*
import math.{Pi, abs, exp, log, sqrt}

class MVNormalTests extends FunSuite:

  // Type aliases for better readability
  type Dim = "dim"
  type Samples = "samples"
  type Feature = "feature"

  // Test tolerance for floating point comparisons
  val tolerance = 1e-4f

  def assertApproxEqual(actual: Float, expected: Float, tolerance: Float = tolerance): Unit =
    if abs(actual - expected) > tolerance then fail(s"Expected $expected, but got $actual (tolerance: $tolerance)")

  override def beforeAll(): Unit =
    super.beforeAll()

  test("MVNormal creation with 1D parameters") {
    val mu = Tensor1[Dim](Seq(0.0f))
    val cov = Tensor2[Dim, Dim](Seq(Seq(1.0f)))
    val mvn = MVNormal(mu, cov)

    assert(mvn != null)
  }

  test("MVNormal creation with 2D parameters") {
    val mu = Tensor1[Dim](Seq(0.0f, 1.0f))
    val cov = Tensor2[Dim, Dim](
      Seq(
        Seq(1.0f, 0.0f),
        Seq(0.0f, 2.0f)
      )
    )
    val mvn = MVNormal(mu, cov)

    assert(mvn != null)
  }

  test("logpdf computation for 1D standard multivariate normal") {
    val mu = Tensor1[Dim](Seq(0.0f))
    val cov = Tensor2[Dim, Dim](Seq(Seq(1.0f)))
    val mvn = MVNormal(mu, cov)

    // Test at x = [0] (should be peak of standard normal)
    val x = Tensor1[Dim](Seq(0.0f))
    val logpdf = mvn.logpdf(x)

    // For 1D standard normal at x=0: -0.5 * log(2π)
    val expected = -0.5f * log(2 * Pi).toFloat
    assertApproxEqual(logpdf.toFloat, expected)
  }

  test("logpdf computation for 1D normal at x=1") {
    val mu = Tensor1[Dim](Seq(0.0f))
    val cov = Tensor2[Dim, Dim](Seq(Seq(1.0f)))
    val mvn = MVNormal(mu, cov)

    val x = Tensor1[Dim](Seq(1.0f))
    val logpdf = mvn.logpdf(x)

    // For 1D standard normal at x=1: -0.5 * log(2π) - 0.5 * 1^2
    val expected = -0.5f * log(2 * Pi).toFloat - 0.5f
    assertApproxEqual(logpdf.toFloat, expected)
  }

  test("logpdf computation for 2D identity covariance") {
    val mu = Tensor1[Dim](Seq(0.0f, 0.0f))
    val cov = Tensor2[Dim, Dim](
      Seq(
        Seq(1.0f, 0.0f),
        Seq(0.0f, 1.0f)
      )
    )
    val mvn = MVNormal(mu, cov)

    // Test at mean
    val x = Tensor1[Dim](Seq(0.0f, 0.0f))
    val logpdf = mvn.logpdf(x)

    // For 2D standard normal at mean: -0.5 * 2 * log(2π) = -log(2π)
    val expected = -log(2 * Pi).toFloat
    assertApproxEqual(logpdf.toFloat, expected)
  }

  test("logpdf computation for 2D diagonal covariance") {
    val mu = Tensor1[Dim](Seq(1.0f, 2.0f))
    val cov = Tensor2[Dim, Dim](
      Seq(
        Seq(4.0f, 0.0f),
        Seq(0.0f, 9.0f)
      )
    )
    val mvn = MVNormal(mu, cov)

    // Test at mean
    val x = Tensor1[Dim](Seq(1.0f, 2.0f))
    val logpdf = mvn.logpdf(x)

    // det(cov) = 4 * 9 = 36
    // logdet = log(36) = log(4*9) = log(4) + log(9)
    val logdet = log(36.0f).toFloat
    val expected = -0.5f * (2.0f * log(2 * Pi).toFloat + logdet)
    assertApproxEqual(logpdf.toFloat, expected)
  }

  test("logpdf symmetry around mean for 2D case") {
    val mu = Tensor1[Dim](Seq(1.0f, 2.0f))
    val cov = Tensor2[Dim, Dim](
      Seq(
        Seq(1.0f, 0.0f),
        Seq(0.0f, 1.0f)
      )
    )
    val mvn = MVNormal(mu, cov)

    val x1 = Tensor1[Dim](Seq(0.0f, 2.0f)) // mu + (-1, 0)
    val x2 = Tensor1[Dim](Seq(2.0f, 2.0f)) // mu + (1, 0)

    val logpdf1 = mvn.logpdf(x1)
    val logpdf2 = mvn.logpdf(x2)

    // Should be equal due to symmetry
    assertApproxEqual(logpdf1.toFloat, logpdf2.toFloat)
  }

  test("sample generation produces correct shape for 1D") {
    val mu = Tensor1[Dim](Seq(0.0f))
    val cov = Tensor2[Dim, Dim](Seq(Seq(1.0f)))
    val mvn = MVNormal(mu, cov)
    val key = shapeful.random.Random.Key(42)

    val sample = mvn.sample(key)

    assertEquals(sample.shape.dims, Seq(1))
    assertEquals(sample.dtype, DType.Float32)
  }

  test("sample generation produces correct shape for 2D") {
    val mu = Tensor1[Dim](Seq(0.0f, 1.0f))
    val cov = Tensor2[Dim, Dim](
      Seq(
        Seq(1.0f, 0.0f),
        Seq(0.0f, 2.0f)
      )
    )
    val mvn = MVNormal(mu, cov)
    val key = shapeful.random.Random.Key(123)

    val sample = mvn.sample(key)

    assertEquals(sample.shape.dims, Seq(2))
    assertEquals(sample.dtype, DType.Float32)
  }

  test("multiple sample generation produces correct shape") {
    val mu = Tensor1[Dim](Seq(0.0f, 1.0f))
    val cov = Tensor2[Dim, Dim](
      Seq(
        Seq(1.0f, 0.0f),
        Seq(0.0f, 2.0f)
      )
    )
    val mvn = MVNormal(mu, cov)
    val key = shapeful.random.Random.Key(456)

    val n = 5
    val samples = mvn.sampleBatch[Samples](n, key)

    assertEquals(samples.shape.dims, Seq(n, 2))
    assertEquals(samples.dtype, DType.Float32)
  }

  test("logpdf decreases with distance from mean") {
    val mu = Tensor1[Dim](Seq(0.0f, 0.0f))
    val cov = Tensor2[Dim, Dim](
      Seq(
        Seq(1.0f, 0.0f),
        Seq(0.0f, 1.0f)
      )
    )
    val mvn = MVNormal(mu, cov)

    val x_close = Tensor1[Dim](Seq(0.1f, 0.1f)) // Close to mean
    val x_far = Tensor1[Dim](Seq(2.0f, 2.0f)) // Far from mean

    val logpdf_close = mvn.logpdf(x_close)
    val logpdf_far = mvn.logpdf(x_far)

    // Closer point should have higher log probability
    assert(logpdf_close.toFloat > logpdf_far.toFloat)
  }

  test("logpdf with positive definite non-diagonal covariance") {
    val mu = Tensor1[Dim](Seq(0.0f, 0.0f))
    val cov = Tensor2[Dim, Dim](
      Seq(
        Seq(2.0f, 0.5f),
        Seq(0.5f, 1.0f)
      )
    )
    val mvn = MVNormal(mu, cov)

    val x = Tensor1[Dim](Seq(0.0f, 0.0f))
    val logpdf = mvn.logpdf(x)

    // Should be finite and negative (since it's a log probability)
    assert(logpdf.toFloat.isFinite)
    assert(logpdf.toFloat < 0.0f)
  }

  test("logpdf numerical stability for extreme values") {
    val mu = Tensor1[Dim](Seq(0.0f, 0.0f))
    val cov = Tensor2[Dim, Dim](
      Seq(
        Seq(1.0f, 0.0f),
        Seq(0.0f, 1.0f)
      )
    )
    val mvn = MVNormal(mu, cov)

    // Test very far from mean
    val x = Tensor1[Dim](Seq(10.0f, 10.0f))
    val logpdf = mvn.logpdf(x)

    // Should be finite but very negative
    assert(logpdf.toFloat.isFinite)
    assert(logpdf.toFloat < -10.0f)
  }

  test("different covariance structures give different logpdf") {
    val mu = Tensor1[Dim](Seq(0.0f, 0.0f))
    val x = Tensor1[Dim](Seq(1.0f, 1.0f))

    // Identity covariance
    val cov1 = Tensor2[Dim, Dim](
      Seq(
        Seq(1.0f, 0.0f),
        Seq(0.0f, 1.0f)
      )
    )
    val mvn1 = MVNormal(mu, cov1)

    // Scaled covariance (higher variance)
    val cov2 = Tensor2[Dim, Dim](
      Seq(
        Seq(4.0f, 0.0f),
        Seq(0.0f, 4.0f)
      )
    )
    val mvn2 = MVNormal(mu, cov2)

    val logpdf1 = mvn1.logpdf(x)
    val logpdf2 = mvn2.logpdf(x)

    // Different covariances should give different log probabilities
    assert(logpdf1.toFloat != logpdf2.toFloat)
    // Higher variance should give lower probability for the same distance
    assert(logpdf2.toFloat < logpdf1.toFloat)
  }

  test("MVNormal reduces to univariate normal for 1D case") {
    val mu_mv = Tensor1[Dim](Seq(2.0f))
    val cov_mv = Tensor2[Dim, Dim](Seq(Seq(9.0f))) // variance = 9, sigma = 3
    val mvn = MVNormal(mu_mv, cov_mv)

    // Equivalent univariate normal
    val mu_uv = Tensor(Shape1[Dim](1), Seq(2.0f), DType.Float32)
    val sigma_uv = Tensor(Shape1[Dim](1), Seq(3.0f), DType.Float32)
    val normal = Normal(mu_uv, sigma_uv)

    val x_mv = Tensor1[Dim](Seq(5.0f))
    val x_uv = Tensor(Shape1[Dim](1), Seq(5.0f), DType.Float32)

    val logpdf_mv = mvn.logpdf(x_mv)
    val logpdf_uv = normal.logpdf(x_uv)

    // Both should be scalars - MVNormal returns joint probability, Normal returns sum of independent log probs
    // Should give approximately the same result
    assertApproxEqual(logpdf_mv.toFloat, logpdf_uv.toFloat, tolerance = 1e-3f)
  }

  test("3D MVNormal basic functionality") {
    val mu = Tensor1[Dim](Seq(0.0f, 1.0f, -1.0f))
    val cov = Tensor2[Dim, Dim](
      Seq(
        Seq(1.0f, 0.0f, 0.0f),
        Seq(0.0f, 2.0f, 0.0f),
        Seq(0.0f, 0.0f, 0.5f)
      )
    )
    val mvn = MVNormal(mu, cov)
    val key = shapeful.random.Random.Key(789)

    // Test sample generation
    val sample = mvn.sample(key)
    assertEquals(sample.shape.dims, Seq(3))

    // Test logpdf at mean
    val logpdf = mvn.logpdf(mu)
    // det(cov) = 1 * 2 * 0.5 = 1
    // logdet = log(1) = 0
    val expected = -0.5f * 3.0f * log(2 * Pi).toFloat
    assertApproxEqual(logpdf.toFloat, expected)
  }
