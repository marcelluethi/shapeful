package shapeful.distributions

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.jax.Jax
import me.shadaj.scalapy.py.SeqConverters

/** Scalar Uniform distribution (single random variable)
  * 
  * Continuous uniform distribution on [low, high).
  * 
  * @param low Lower bound (inclusive)
  * @param high Upper bound (exclusive)
  */
class ScalarUniform(val low: Tensor0, val high: Tensor0) extends ScalarDistribution:
  type Support = Tensor0
  
  /** Log probability density function */
  def logpdf(x: Tensor0): Tensor0 =
    val scale = high - low
    val logp_value = Jax.scipy_stats.uniform.logpdf(
      x.jaxValue,
      loc = low.jaxValue,
      scale = scale.jaxValue
    )
    new Tensor[EmptyTuple](Shape.empty, logp_value, x.dtype)
  
  /** Sample from Uniform distribution */
  def sample(key: shapeful.random.Random.Key): Tensor0 =
    // JAX's uniform is on [0, 1), so we transform
    val unif_sample = Jax.jrandom.uniform(key.jaxKey, minval = low.jaxValue, maxval = high.jaxValue)
    new Tensor[EmptyTuple](Shape.empty, unif_sample, DType.Float32)

/** Uniform distribution
  * 
  * Continuous uniform distribution with independent elements on [low, high).
  * 
  * @param low Lower bound for each element (inclusive)
  * @param high Upper bound for each element (exclusive)
  */
class Uniform[S <: Tuple](val low: Tensor[S], val high: Tensor[S]) extends FactorizedDistribution[S]:

  def shape: Shape[S] = low.shape

  /** Element-wise log probability density function */
  def logpdfElements(x: Tensor[S]): Tensor[S] =
    val scale = high - low
    val logp_values = Jax.scipy_stats.uniform.logpdf(
      x.jaxValue,
      loc = low.jaxValue,
      scale = scale.jaxValue
    )
    new Tensor[S](x.shape, logp_values, x.dtype)

  /** Generate samples from the Uniform distribution */
  def sample(key: shapeful.random.Random.Key): Tensor[S] =
    // JAX's uniform can take minval/maxval tensors directly
    val unif_sample = Jax.jrandom.uniform(
      key.jaxKey, 
      low.shape.dims.toPythonProxy,
      minval = low.jaxValue, 
      maxval = high.jaxValue
    )
    new Tensor[S](low.shape, unif_sample, low.dtype)
  
  /** Mean of Uniform: (low + high) / 2 */
  def mean: Tensor[S] = (low + high) * Tensor0(0.5f)
  
  /** Variance of Uniform: (high - low)² / 12 */
  def variance: Tensor[S] =
    val range = high - low
    range.pow(Tensor0(2f)) * Tensor0(1.0f / 12.0f)

object Uniform:

  /** Create a factorized Uniform distribution */
  def apply[S <: Tuple](low: Tensor[S], high: Tensor[S]): Uniform[S] =
    new Uniform(low, high)

  /** Create a scalar Uniform distribution */
  def scalar(low: Tensor0, high: Tensor0): ScalarUniform =
    new ScalarUniform(low, high)

  /** Standard uniform on [0, 1) */
  def standard[S <: Tuple](shape: Shape[S]): Uniform[S] =
    new Uniform(Tensor.zeros(shape), Tensor.ones(shape))
  
  /** Standard scalar uniform on [0, 1) */
  def standardScalar: ScalarUniform =
    new ScalarUniform(Tensor0(0.0f), Tensor0(1.0f))
