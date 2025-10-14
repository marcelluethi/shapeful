package shapeful.distributions

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.jax.Jax

/** Scalar Beta distribution (single random variable)
  *
  * Continuous distribution on [0, 1], commonly used for modeling probabilities.
  *
  * @param alpha
  *   Shape parameter α (must be positive)
  * @param beta
  *   Shape parameter β (must be positive)
  */
class ScalarBeta(val alpha: Tensor0, val beta: Tensor0) extends ScalarDistribution:
  type Support = Tensor0

  /** Log probability density function */
  def logpdf(x: Tensor0): Tensor0 =
    val logp_value = Jax.scipy_stats.beta.logpdf(
      x.jaxValue,
      a = alpha.jaxValue,
      b = beta.jaxValue
    )
    new Tensor[EmptyTuple](Shape.empty, logp_value, x.dtype)

  /** Sample from Beta distribution */
  def sample(key: shapeful.random.Random.Key): Tensor0 =
    // Use JAX's beta sampler
    val sample_value = Jax.jrandom.beta(
      key.jaxKey,
      a = alpha.jaxValue,
      b = beta.jaxValue
    )
    new Tensor[EmptyTuple](Shape.empty, sample_value, DType.Float32)

/** Beta distribution
  *
  * Continuous distribution on [0, 1] with independent elements. Commonly used for modeling probabilities and
  * proportions.
  *
  * @param alpha
  *   Shape parameter α for each element (must be positive)
  * @param beta
  *   Shape parameter β for each element (must be positive)
  */
class Beta[S <: Tuple](val alpha: Tensor[S], val beta: Tensor[S]) extends FactorizedDistribution[S]:

  def shape: Shape[S] = alpha.shape

  /** Element-wise log probability density function */
  def logpdfElements(x: Tensor[S]): Tensor[S] =
    val logp_values = Jax.scipy_stats.beta.logpdf(
      x.jaxValue,
      a = alpha.jaxValue,
      b = beta.jaxValue
    )
    new Tensor[S](x.shape, logp_values, x.dtype)

  /** Generate samples from the Beta distribution */
  def sample(key: shapeful.random.Random.Key): Tensor[S] =
    val samples = Jax.jrandom.beta(
      key.jaxKey,
      a = alpha.jaxValue,
      b = beta.jaxValue
    )
    new Tensor[S](alpha.shape, samples, alpha.dtype)

  /** Mean of Beta: α / (α + β) */
  def mean: Tensor[S] = alpha / (alpha + beta)

  /** Variance of Beta: αβ / ((α+β)²(α+β+1)) */
  def variance: Tensor[S] =
    val sum = alpha + beta
    val numerator = alpha * beta
    val denominator = sum.pow(Tensor0(2f)) * (sum + Tensor.ones(alpha.shape))
    numerator / denominator

object Beta:

  /** Create a factorized Beta distribution */
  def apply[S <: Tuple](alpha: Tensor[S], beta: Tensor[S]): Beta[S] =
    new Beta(alpha, beta)

  /** Create a scalar Beta distribution */
  def scalar(alpha: Tensor0, beta: Tensor0): ScalarBeta =
    new ScalarBeta(alpha, beta)

  /** Uniform distribution on [0,1] (α = β = 1) */
  def uniform[S <: Tuple](shape: Shape[S]): Beta[S] =
    new Beta(Tensor.ones(shape), Tensor.ones(shape))

  /** Uniform scalar on [0,1] */
  def uniformScalar: ScalarBeta =
    new ScalarBeta(Tensor0(1.0f), Tensor0(1.0f))
