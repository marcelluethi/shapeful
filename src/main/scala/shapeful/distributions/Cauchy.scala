package shapeful.distributions

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.jax.Jax
import me.shadaj.scalapy.py.SeqConverters

/** Scalar Cauchy distribution (single random variable)
  * 
  * Heavy-tailed distribution with no defined mean or variance.
  * Also known as the Lorentz distribution.
  * 
  * @param loc Location parameter (median)
  * @param scale Scale parameter (half-width at half-maximum, must be positive)
  */
class ScalarCauchy(val loc: Tensor0, val scale: Tensor0) extends ScalarDistribution:
  type Support = Tensor0
  
  /** Log probability density function */
  def logpdf(x: Tensor0): Tensor0 =
    val logp_value = Jax.scipy_stats.cauchy.logpdf(
      x.jaxValue,
      loc = loc.jaxValue,
      scale = scale.jaxValue
    )
    new Tensor[EmptyTuple](Shape.empty, logp_value, x.dtype)
  
  /** Sample from Cauchy distribution */
  def sample(key: shapeful.random.Random.Key): Tensor0 =
    // JAX's cauchy is standard (loc=0, scale=1), so we transform
    val cauchy_sample = Jax.jrandom.cauchy(key.jaxKey)
    loc + scale * new Tensor[EmptyTuple](Shape.empty, cauchy_sample, DType.Float32)

/** Cauchy distribution
  * 
  * Heavy-tailed distribution with independent elements.
  * Has no defined mean or variance (infinite).
  * 
  * @param loc Location parameter (median) for each element
  * @param scale Scale parameter for each element (must be positive)
  */
class Cauchy[S <: Tuple](val loc: Tensor[S], val scale: Tensor[S]) extends FactorizedDistribution[S]:

  def shape: Shape[S] = loc.shape

  /** Element-wise log probability density function */
  def logpdfElements(x: Tensor[S]): Tensor[S] =
    val logp_values = Jax.scipy_stats.cauchy.logpdf(
      x.jaxValue,
      loc = loc.jaxValue,
      scale = scale.jaxValue
    )
    new Tensor[S](x.shape, logp_values, x.dtype)

  /** Generate samples from the Cauchy distribution */
  def sample(key: shapeful.random.Random.Key): Tensor[S] =
    // JAX's cauchy is standard (loc=0, scale=1), so we transform
    val cauchy_sample = Jax.jrandom.cauchy(key.jaxKey, loc.shape.dims.toPythonProxy)
    loc + scale * new Tensor[S](loc.shape, cauchy_sample, loc.dtype)
  
  /** Mean of Cauchy: undefined (infinite) - returns NaN */
  def mean: Tensor[S] = 
    Tensor.ones(loc.shape) * Tensor0(Float.NaN)
  
  /** Variance of Cauchy: undefined (infinite) - returns NaN */
  def variance: Tensor[S] = 
    Tensor.ones(loc.shape) * Tensor0(Float.NaN)

object Cauchy:

  /** Create a factorized Cauchy distribution */
  def apply[S <: Tuple](loc: Tensor[S], scale: Tensor[S]): Cauchy[S] =
    new Cauchy(loc, scale)

  /** Create a scalar Cauchy distribution */
  def scalar(loc: Tensor0, scale: Tensor0): ScalarCauchy =
    new ScalarCauchy(loc, scale)

  /** Standard Cauchy (loc=0, scale=1) */
  def standard[S <: Tuple](shape: Shape[S]): Cauchy[S] =
    new Cauchy(Tensor.zeros(shape), Tensor.ones(shape))
  
  /** Standard scalar Cauchy */
  def standardScalar: ScalarCauchy =
    new ScalarCauchy(Tensor0(0.0f), Tensor0(1.0f))
