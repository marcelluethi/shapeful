package shapeful.distributions

import scala.language.experimental.namedTypeArguments
import math.Pi
import shapeful.*
import shapeful.jax.Jax

/** Scalar half-normal distribution (single random variable)
  * 
  * Represents a single half-normal random variable (support on [0, ∞))
  * The distribution of |X| where X ~ Normal(0, σ)
  * 
  * @param sigma Scale parameter (must be positive)
  */
class ScalarHalfNormal(val sigma: Tensor0) extends ScalarDistribution:
  type Support = Tensor0
  
  /** Log probability density for x >= 0 */
  def logpdf(x: Tensor0): Tensor0 =
    val logpdf_value = Jax.scipy_stats.halfnorm.logpdf(
      x.jaxValue,
      loc = 0.0,
      scale = sigma.jaxValue
    )
    new Tensor[EmptyTuple](Shape.empty, logpdf_value, x.dtype)
  
  /** Sample from half-normal distribution */
  def sample(key: shapeful.random.Random.Key): Tensor0 =
    val z = Tensor.randn(key, Shape.empty, DType.Float32)
    (z * sigma).abs

/** Half-Normal distribution
  * 
  * The half-normal distribution is the distribution of |X| where X ~ Normal(0, σ).
  * It only has support on [0, ∞).
  * 
  * This is a factorized distribution: each element is an independent half-normal random variable.
  * 
  * Uses scipy.stats.halfnorm for efficient computation via JAX.
  * 
  * @param sigma Scale parameter (must be positive)
  */
class HalfNormal[S <: Tuple](val sigma: Tensor[S]) extends FactorizedDistribution[S]:

  def shape: Shape[S] = sigma.shape

  /** Element-wise log probabilities for x >= 0
    *
    * Uses scipy.stats.halfnorm.logpdf
    *
    * @param x Input tensor (must be non-negative)
    * @return log(PDF) values for each element
    */
  def logpdfElements(x: Tensor[S]): Tensor[S] =
    val logpdf_values = Jax.scipy_stats.halfnorm.logpdf(
      x.jaxValue,
      loc = 0.0,
      scale = sigma.jaxValue
    )
    new Tensor[S](x.shape, logpdf_values, x.dtype)

  /** Generate samples from the half-normal distribution
    *
    * @param key Random key for sampling
    * @return Samples from the half-normal distribution (non-negative)
    */
  def sample(key: shapeful.random.Random.Key): Tensor[S] =
    // Sample from Normal(0, σ) and take absolute value
    val z = Tensor.randn(key, sigma.shape, sigma.dtype)
    (z * sigma).abs
  
  /** Mean of the half-normal distribution: σ * sqrt(2/π) */
  def mean: Tensor[S] =
    sigma * Tensor0(math.sqrt(2.0 / Pi).toFloat)
  
  /** Variance of the half-normal distribution: σ²(1 - 2/π) */
  def variance: Tensor[S] =
    val sigmaSq = sigma.pow(Tensor0(2f))
    sigmaSq * Tensor0((1.0 - 2.0 / Pi).toFloat)

object HalfNormal:

  /** Create a factorized half-normal distribution with custom σ */
  def apply[S <: Tuple](sigma: Tensor[S]): HalfNormal[S] =
    new HalfNormal(sigma)

  /** Create a scalar half-normal distribution */
  def scalar(sigma: Tensor0): ScalarHalfNormal =
    new ScalarHalfNormal(sigma)

  /** Create a standard half-normal distribution with σ = 1 */
  def standard[S <: Tuple](shape: Shape[S]): HalfNormal[S] =
    new HalfNormal(Tensor.ones(shape))

  /** Create a standard scalar half-normal with σ = 1 */
  def standardScalar: ScalarHalfNormal =
    new ScalarHalfNormal(Tensor0(1f))

  /** Create a half-normal distribution with scalar σ broadcast to given shape */
  def apply[S <: Tuple](sigma: Float, shape: Shape[S]): HalfNormal[S] =
    new HalfNormal(Tensor.ones(shape) * Tensor0(sigma))
