package shapeful.distributions

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.jax.Jax
import me.shadaj.scalapy.py.SeqConverters

/** Scalar Exponential distribution (single random variable)
  * 
  * Models time between events in a Poisson process.
  * Support on [0, ∞)
  * 
  * @param rate Rate parameter λ (must be positive)
  */
class ScalarExponential(val rate: Tensor0) extends ScalarDistribution:
  type Support = Tensor0
  
  /** Log probability density function */
  def logpdf(x: Tensor0): Tensor0 =
    val scale = Tensor0(1.0f) / rate
    val logp_value = Jax.scipy_stats.expon.logpdf(
      x.jaxValue,
      scale = scale.jaxValue
    )
    new Tensor[EmptyTuple](Shape.empty, logp_value, x.dtype)
  
  /** Sample from Exponential distribution */
  def sample(key: shapeful.random.Random.Key): Tensor0 =
    // JAX's exponential has rate=1, so we scale by 1/rate
    val exp_sample = Jax.jrandom.exponential(key.jaxKey)
    new Tensor[EmptyTuple](Shape.empty, exp_sample, DType.Float32) / rate

/** Exponential distribution
  * 
  * Models time between events with independent elements.
  * Support on [0, ∞)
  * 
  * @param rate Rate parameter λ for each element (must be positive)
  */
class Exponential[S <: Tuple](val rate: Tensor[S]) extends FactorizedDistribution[S]:

  def shape: Shape[S] = rate.shape

  /** Element-wise log probability density function */
  def logpdfElements(x: Tensor[S]): Tensor[S] =
    val scale = Tensor.ones(rate.shape) / rate  // scale = 1/λ
    val logp_values = Jax.scipy_stats.expon.logpdf(
      x.jaxValue,
      scale = scale.jaxValue
    )
    new Tensor[S](x.shape, logp_values, x.dtype)

  /** Generate samples from the Exponential distribution */
  def sample(key: shapeful.random.Random.Key): Tensor[S] =
    // JAX's exponential has rate=1, so we scale by 1/rate
    val exp_sample = Jax.jrandom.exponential(key.jaxKey, rate.shape.dims.toPythonProxy)
    new Tensor[S](rate.shape, exp_sample, rate.dtype) / rate
  
  /** Mean of Exponential: 1/λ */
  def mean: Tensor[S] = Tensor.ones(rate.shape) / rate
  
  /** Variance of Exponential: 1/λ² */
  def variance: Tensor[S] =
    val rateSq = rate.pow(Tensor0(2f))
    Tensor.ones(rate.shape) / rateSq

object Exponential:

  /** Create a factorized Exponential distribution */
  def apply[S <: Tuple](rate: Tensor[S]): Exponential[S] =
    new Exponential(rate)

  /** Create a scalar Exponential distribution */
  def scalar(rate: Tensor0): ScalarExponential =
    new ScalarExponential(rate)

  /** Unit rate exponential (λ = 1) */
  def standard[S <: Tuple](shape: Shape[S]): Exponential[S] =
    new Exponential(Tensor.ones(shape))
  
  /** Unit rate scalar exponential */
  def standardScalar: ScalarExponential =
    new ScalarExponential(Tensor0(1.0f))
