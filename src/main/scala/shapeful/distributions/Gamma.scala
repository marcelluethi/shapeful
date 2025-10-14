package shapeful.distributions

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.jax.Jax
import me.shadaj.scalapy.py.SeqConverters

/** Scalar Gamma distribution (single random variable)
  *
  * Continuous distribution on (0, ∞), generalizes exponential distribution.
  *
  * @param alpha
  *   Shape parameter α (must be positive)
  * @param beta
  *   Rate parameter β (must be positive)
  */
class ScalarGamma(val alpha: Tensor0, val beta: Tensor0) extends ScalarDistribution:
  type Support = Tensor0

  /** Log probability density function */
  def logpdf(x: Tensor0): Tensor0 =
    val scale = Tensor0(1.0f) / beta
    val logp_value = Jax.scipy_stats.gamma.logpdf(
      x.jaxValue,
      a = alpha.jaxValue,
      scale = scale.jaxValue
    )
    new Tensor[EmptyTuple](Shape.empty, logp_value, x.dtype)

  /** Sample from Gamma distribution */
  def sample(key: shapeful.random.Random.Key): Tensor0 =
    val sample_value = Jax.jrandom.gamma(
      key.jaxKey,
      alpha.jaxValue,
      Seq[Int]().toPythonProxy
    )
    val scale = Tensor0(1.0f) / beta
    new Tensor[EmptyTuple](Shape.empty, sample_value, DType.Float32) * scale

/** Gamma distribution
  *
  * Continuous distribution on (0, ∞) with independent elements.
  *
  * @param alpha
  *   Shape parameter for each element (must be positive)
  * @param beta
  *   Rate parameter for each element (must be positive)
  */
class Gamma[S <: Tuple](val alpha: Tensor[S], val beta: Tensor[S]) extends FactorizedDistribution[S]:

  def shape: Shape[S] = alpha.shape

  /** Element-wise log probability density function */
  def logpdfElements(x: Tensor[S]): Tensor[S] =
    val scale = Tensor.ones(beta.shape) / beta
    val logp_values = Jax.scipy_stats.gamma.logpdf(
      x.jaxValue,
      a = alpha.jaxValue,
      scale = scale.jaxValue
    )
    new Tensor[S](x.shape, logp_values, x.dtype)

  /** Generate samples from the Gamma distribution */
  def sample(key: shapeful.random.Random.Key): Tensor[S] =
    val samples = Jax.jrandom.gamma(
      key.jaxKey,
      a = alpha.jaxValue,
      shape = alpha.shape.dims.toPythonProxy
    )
    new Tensor[S](alpha.shape, samples, alpha.dtype) / beta

  /** Mean of Gamma: α / β */
  def mean: Tensor[S] = alpha / beta

  /** Variance of Gamma: α / β² */
  def variance: Tensor[S] =
    val betaSq = beta.pow(Tensor0(2f))
    alpha / betaSq

object Gamma:

  /** Create a factorized Gamma distribution */
  def apply[S <: Tuple](alpha: Tensor[S], beta: Tensor[S]): Gamma[S] =
    new Gamma(alpha, beta)

  /** Create a scalar Gamma distribution */
  def scalar(alpha: Tensor0, beta: Tensor0): ScalarGamma =
    new ScalarGamma(alpha, beta)

  /** Standard gamma with α = 1 (equivalent to exponential with rate β) */
  def exponential[S <: Tuple](beta: Tensor[S]): Gamma[S] =
    new Gamma(Tensor.ones(beta.shape), beta)

  /** Chi-squared distribution with k degrees of freedom */
  def chiSquared[S <: Tuple](k: Tensor[S]): Gamma[S] =
    new Gamma(k * Tensor0(0.5f), Tensor.ones(k.shape) * Tensor0(0.5f))
