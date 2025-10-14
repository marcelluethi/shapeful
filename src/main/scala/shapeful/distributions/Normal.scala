package shapeful.distributions

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.jax.Jax

/** Scalar normal distribution (single random variable)
  * 
  * Represents a single Gaussian random variable N(μ, σ²)
  * 
  * @param mu Mean of the distribution
  * @param sigma Standard deviation (must be positive)
  */
class ScalarNormal(val mu: Tensor0, val sigma: Tensor0) extends ScalarDistribution:
  type Support = Tensor0
  
  /** Log probability density */
  def logpdf(x: Tensor0): Tensor0 =
    val logpdf_value = Jax.scipy_stats.norm.logpdf(
      x.jaxValue,
      loc = mu.jaxValue,
      scale = sigma.jaxValue
    )
    new Tensor[EmptyTuple](Shape.empty, logpdf_value, x.dtype)
  
  /** Sample from N(μ, σ²) */
  def sample(key: shapeful.random.Random.Key): Tensor0 =
    val z = Tensor.randn(key, Shape.empty, DType.Float32)
    mu + sigma * z

/** Normal (Gaussian) distribution
  *
  * Wraps JAX's scipy.stats.norm for efficient computation.
  * 
  * This is a factorized distribution: each element of the tensor
  * is an independent normal random variable with its own mean and std dev.
  *
  * @param mu Mean of the distribution
  * @param sigma Standard deviation of the distribution (must be positive)
  */
class Normal[S <: Tuple](val mu: Tensor[S], val sigma: Tensor[S]) 
  extends FactorizedDistribution[S]:

  def shape: Shape[S] = mu.shape

  /** Element-wise log probabilities
    * 
    * Uses JAX's scipy.stats.norm.logpdf
    *
    * @param x Input tensor
    * @return log(PDF) values for each element
    */
  def logpdfElements(x: Tensor[S]): Tensor[S] =
    val logpdf_values = Jax.scipy_stats.norm.logpdf(
      x.jaxValue,
      loc = mu.jaxValue,
      scale = sigma.jaxValue
    )
    new Tensor[S](x.shape, logpdf_values, x.dtype)

  /** Generate samples from the normal distribution
    *
    * Uses JAX's random.normal for efficient sampling
    *
    * @param key Random key for sampling
    * @return Samples from the normal distribution
    */
  def sample(key: shapeful.random.Random.Key): Tensor[S] =
    val z = Tensor.randn(key, mu.shape, mu.dtype)
    mu + sigma * z

  /** Mean of the distribution */
  def mean: Tensor[S] = mu

  /** Variance of the distribution */
  def variance: Tensor[S] = sigma.pow(Tensor0(2f))

  /** Standard deviation of the distribution */
  def std: Tensor[S] = sigma

object Normal:
  /** Create a factorized normal distribution with specified means and standard deviations */
  def apply[S <: Tuple](mu: Tensor[S], sigma: Tensor[S]): Normal[S] =
    new Normal(mu, sigma)

  /** Create a scalar normal distribution N(μ, σ²) */
  def scalar(mu: Tensor0, sigma: Tensor0): ScalarNormal =
    new ScalarNormal(mu, sigma)

  /** Standard normal distribution N(0, 1) with given shape */
  def standard[S <: Tuple](shape: Shape[S]): Normal[S] =
    new Normal(Tensor.zeros(shape), Tensor.ones(shape))
  
  /** Standard scalar normal N(0, 1) */
  def standardScalar: ScalarNormal =
    new ScalarNormal(Tensor0(0f), Tensor0(1f))
