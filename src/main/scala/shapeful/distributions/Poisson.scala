package shapeful.distributions

import shapeful.*
import shapeful.jax.Jax
import me.shadaj.scalapy.py.SeqConverters

/** Scalar Poisson distribution (single random variable)
  *
  * Discrete distribution for modeling counts (non-negative integers). Models the number of events occurring in a fixed
  * interval.
  *
  * @param rate
  *   Rate parameter λ (must be positive)
  */
class ScalarPoisson(val rate: Tensor0) extends ScalarDistribution:
  type Support = Tensor0

  /** Log probability mass function */
  def logpdf(x: Tensor0): Tensor0 =
    val logp_value = Jax.scipy_stats.poisson.logpmf(
      x.jaxValue,
      mu = rate.jaxValue
    )
    new Tensor[EmptyTuple](Shape.empty, logp_value, x.dtype)

  /** Sample from Poisson distribution */
  def sample(key: shapeful.random.Random.Key): Tensor0 =
    val sample_value = Jax.jrandom.poisson(
      key.jaxKey,
      lam = rate.jaxValue
    )
    new Tensor[EmptyTuple](Shape.empty, sample_value, DType.Float32)

/** Poisson distribution
  *
  * Discrete distribution for modeling counts with independent elements. Support on non-negative integers.
  *
  * @param rate
  *   Rate parameter λ for each element (must be positive)
  */
class Poisson[S <: Tuple](val rate: Tensor[S]) extends FactorizedDistribution[S]:

  def shape: Shape[S] = rate.shape

  /** Element-wise log probability mass function */
  def logpdfElements(x: Tensor[S]): Tensor[S] =
    val logp_values = Jax.scipy_stats.poisson.logpmf(
      x.jaxValue,
      mu = rate.jaxValue
    )
    new Tensor[S](x.shape, logp_values, x.dtype)

  /** Generate samples from the Poisson distribution */
  def sample(key: shapeful.random.Random.Key): Tensor[S] =
    val samples = Jax.jrandom.poisson(
      key.jaxKey,
      lam = rate.jaxValue,
      shape = rate.shape.dims.toPythonProxy
    )
    new Tensor[S](rate.shape, samples, rate.dtype)

  /** Mean of Poisson: λ */
  def mean: Tensor[S] = rate

  /** Variance of Poisson: λ */
  def variance: Tensor[S] = rate

object Poisson:

  /** Create a factorized Poisson distribution */
  def apply[S <: Tuple](rate: Tensor[S]): Poisson[S] =
    new Poisson(rate)

  /** Create a scalar Poisson distribution */
  def scalar(rate: Tensor0): ScalarPoisson =
    new ScalarPoisson(rate)

  /** Unit rate Poisson (λ = 1) */
  def standard[S <: Tuple](shape: Shape[S]): Poisson[S] =
    new Poisson(Tensor.ones(shape))
