package shapeful.distributions

import scala.language.experimental.namedTypeArguments
import math.Pi
import shapeful.*

class Normal[S <: Tuple](mu: Tensor[S], sigma: Tensor[S]):

  // Precompute constants for efficiency
  private val variance = sigma.pow(Tensor0(2))
  private val logSigma = sigma.log
  private val log2Pi = Tensor0(math.log(2 * Pi).toFloat)

  /** Computes the log of the probability density function at x More numerically stable for very small probabilities
    *
    * @param x
    *   Input tensor
    * @return
    *   log(PDF) values for each input
    */
  def logpdf(x: Tensor[S]): Tensor[S] =
    // Log PDF formula: -0.5 * log(2π) - log(σ) - 0.5 * ((x - μ)/σ)^2
    val term1 = log2Pi * Tensor0(-0.5f)
    val term2 = logSigma * Tensor0(-1.0f)
    val xsubmu = x - mu
    val term3 = (xsubmu.pow(Tensor0(2f)) / variance) * Tensor0(-0.5f)  // Fixed!
    (term2 + term3) + term1

  /** Generate samples from the normal distribution
    *
    * @return
    *   Samples from the normal distribution
    */
  def sample(): Tensor[S] =
    val z = Tensor.randn(mu.shape)
    mu + sigma * z


