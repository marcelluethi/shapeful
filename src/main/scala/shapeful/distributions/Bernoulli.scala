package shapeful.distributions

import shapeful.*
import shapeful.jax.Jax
import me.shadaj.scalapy.py.SeqConverters

/** Scalar Bernoulli distribution (single random variable)
  *
  * Binary distribution: P(X=1) = p, P(X=0) = 1-p
  *
  * @param p
  *   Probability of success (must be in [0, 1])
  */
class ScalarBernoulli(val p: Tensor0) extends ScalarDistribution:
  type Support = Tensor0

  /** Log probability mass function */
  def logpdf(x: Tensor0): Tensor0 =
    val logp_value = Jax.scipy_stats.bernoulli.logpmf(
      x.jaxValue,
      p = p.jaxValue
    )
    new Tensor[EmptyTuple](Shape.empty, logp_value, x.dtype)

  /** Sample from Bernoulli distribution */
  def sample(key: shapeful.random.Random.Key): Tensor0 =
    val bern_sample = Jax.jrandom.bernoulli(key.jaxKey, p.jaxValue)
    new Tensor[EmptyTuple](Shape.empty, bern_sample, DType.Float32)

/** Bernoulli distribution
  *
  * Binary distribution with independent elements. Each element: P(X=1) = p, P(X=0) = 1-p
  *
  * @param p
  *   Probability of success for each element (must be in [0, 1])
  */
class Bernoulli[S <: Tuple](val p: Tensor[S]) extends FactorizedDistribution[S]:

  def shape: Shape[S] = p.shape

  /** Element-wise log probability mass function */
  def logpdfElements(x: Tensor[S]): Tensor[S] =
    val logp_values = Jax.scipy_stats.bernoulli.logpmf(
      x.jaxValue,
      p = p.jaxValue
    )
    new Tensor[S](x.shape, logp_values, x.dtype)

  /** Generate samples from the Bernoulli distribution */
  def sample(key: shapeful.random.Random.Key): Tensor[S] =
    val bern_sample = Jax.jrandom.bernoulli(key.jaxKey, p.jaxValue, p.shape.dims.toPythonProxy)
    new Tensor[S](p.shape, bern_sample, p.dtype)

  /** Mean of Bernoulli: p */
  def mean: Tensor[S] = p

  /** Variance of Bernoulli: p(1-p) */
  def variance: Tensor[S] =
    p * (Tensor.ones(p.shape) - p)

object Bernoulli:

  /** Create a factorized Bernoulli distribution */
  def apply[S <: Tuple](p: Tensor[S]): Bernoulli[S] =
    new Bernoulli(p)

  /** Create a scalar Bernoulli distribution */
  def scalar(p: Tensor0): ScalarBernoulli =
    new ScalarBernoulli(p)

  /** Fair coin flip (p = 0.5) */
  def fair[S <: Tuple](shape: Shape[S]): Bernoulli[S] =
    new Bernoulli(Tensor.ones(shape) * Tensor0(0.5f))

  /** Fair scalar coin flip */
  def fairScalar: ScalarBernoulli =
    new ScalarBernoulli(Tensor0(0.5f))
