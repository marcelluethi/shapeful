package shapeful.distributions

import scala.language.experimental.namedTypeArguments
import math.Pi
import shapeful.*

class HalfNormal[S <: Tuple](sigma: Tensor[S]):

  // Create a Normal(0, sigma) distribution to use internally
  private val normal = Normal(Tensor.zeros(sigma.shape), sigma)

  // Precompute constants for efficiency
  private val log2 = Tensor0(math.log(2).toFloat)
  private val logSigma = sigma.log
  private val log2Pi = Tensor0(math.log(2 * Pi).toFloat)

  /** Computes the log of the probability density function at x for x >= 0 More numerically stable for very small
    * probabilities
    *
    * @param x
    *   Input tensor (must be non-negative)
    * @return
    *   log(PDF) values for each input
    */
  def logpdf(x: Tensor[S]): Tensor[S] =
    // Half-normal log PDF formula for x >= 0:
    // log(2) + Normal(0, σ).logpdf(x)
    // This simplifies to: log(2) - 0.5 * log(2π) - log(σ) - 0.5 * (x/σ)^2
    val normalLogPdf = normal.logpdf(x)
    normalLogPdf + log2

  /** Generate samples from the half-normal distribution
    *
    * @return
    *   Samples from the half-normal distribution (non-negative)
    */
  def sample(key: shapeful.random.Random.Key): Tensor[S] =
    // Sample from Normal(0, σ) and take absolute value
    normal.sample(key).abs

object HalfNormal:

  /** Create a standard half-normal distribution with σ = 1 */
  def standard[S <: Tuple](shape: Shape[S]): HalfNormal[S] =
    new HalfNormal(Tensor.ones(shape))

  /** Create a half-normal distribution with custom σ */
  def apply[S <: Tuple](sigma: Tensor[S]): HalfNormal[S] =
    new HalfNormal(sigma)

  /** Create a half-normal distribution with scalar σ */
  def apply[S <: Tuple](sigma: Float, shape: Shape[S]): HalfNormal[S] =
    new HalfNormal(Tensor.ones(shape) * Tensor0(sigma))
