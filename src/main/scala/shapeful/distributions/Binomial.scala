package shapeful.distributions

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.jax.Jax

/** Scalar Binomial distribution (single random variable)
  * 
  * Number of successes in n independent Bernoulli trials.
  * 
  * @param n Number of trials (must be positive integer)
  * @param p Probability of success per trial (must be in [0, 1])
  */
class ScalarBinomial(val n: Tensor0, val p: Tensor0) extends ScalarDistribution:
  type Support = Tensor0
  
  /** Log probability mass function */
  def logpdf(x: Tensor0): Tensor0 =
    val logp_value = Jax.scipy_stats.binom.logpmf(
      x.jaxValue,
      n = n.jaxValue,
      p = p.jaxValue
    )
    new Tensor[EmptyTuple](Shape.empty, logp_value, x.dtype)
  
  /** Sample from Binomial distribution */
  def sample(key: shapeful.random.Random.Key): Tensor0 =
    val sample_value = Jax.jrandom.binomial(
      key.jaxKey,
      n = n.jaxValue,
      p = p.jaxValue
    )
    new Tensor[EmptyTuple](Shape.empty, sample_value, DType.Float32)

/** Binomial distribution
  * 
  * Number of successes in n independent Bernoulli trials.
  * Each element has independent binomial draws.
  * 
  * @param n Number of trials for each element
  * @param p Probability of success per trial for each element
  */
class Binomial[S <: Tuple](val n: Tensor[S], val p: Tensor[S]) extends FactorizedDistribution[S]:

  def shape: Shape[S] = p.shape

  /** Element-wise log probability mass function */
  def logpdfElements(x: Tensor[S]): Tensor[S] =
    val logp_values = Jax.scipy_stats.binom.logpmf(
      x.jaxValue,
      n = n.jaxValue,
      p = p.jaxValue
    )
    new Tensor[S](x.shape, logp_values, x.dtype)

  /** Generate samples from the Binomial distribution */
  def sample(key: shapeful.random.Random.Key): Tensor[S] =
    // Use JAX's binomial sampler
    val samples = Jax.jrandom.binomial(
      key.jaxKey,
      n = n.jaxValue,
      p = p.jaxValue
    )
    new Tensor[S](n.shape, samples, n.dtype)
  
  /** Mean of Binomial: n * p */
  def mean: Tensor[S] = n * p
  
  /** Variance of Binomial: n * p * (1-p) */
  def variance: Tensor[S] =
    n * p * (Tensor.ones(p.shape) - p)

object Binomial:

  /** Create a factorized Binomial distribution */
  def apply[S <: Tuple](n: Tensor[S], p: Tensor[S]): Binomial[S] =
    new Binomial(n, p)

  /** Create a scalar Binomial distribution */
  def scalar(n: Tensor0, p: Tensor0): ScalarBinomial =
    new ScalarBinomial(n, p)

  /** Create binomial with constant n and varying p */
  def apply[S <: Tuple](n: Int, p: Tensor[S]): Binomial[S] =
    val nTensor = Tensor.ones(p.shape) * Tensor0(n.toFloat)
    new Binomial(nTensor, p)
