package shapeful.distributions


import shapeful.tensor.Tensor
import shapeful.tensor.Tensor.{Tensor1, Tensor0}
import shapeful.tensor.{pow, log, multScalar, sub, div, add, divScalar, addScalar}

import scala.math.{Pi, exp, log, sqrt}

class NormalDistribution[A <: Tuple](mu: Tensor[A], sigma: Tensor[A]) {
  // Precompute constants for efficiency
  private val variance = sigma.pow(2)
  private val logSigma = sigma.log
  private val log2Pi : Tensor0 = Tensor(math.log(2 * Pi).toFloat, requiresGrad = false)


  /**
   * Computes the log of the probability density function at x
   * More numerically stable for very small probabilities
   *
   * @param x Input tensor
   * @return log(PDF) values for each input
   */
  def logpdf(x: Tensor[A]): Tensor[A] = {
    // Log PDF formula: -0.5 * log(2π) - log(σ) - 0.5 * ((x - μ)/σ)^2
    val term1 = log2Pi.multScalar(Tensor(-0.5f, requiresGrad = false))
    val term2 = logSigma.multScalar(Tensor(-1f, requiresGrad = false))
    val term3 = x.sub(mu).pow(2).div(variance).multScalar(Tensor(-0.5f, requiresGrad = false))

    term2.addScalar(term1).add(term3)
  }

  /**
   * Generate samples from the normal distribution
   *
   * @param sampleShape Shape of samples to generate
   * @return Samples from the normal distribution
   */
//  def sample(sampleShape: Array[Int]): Tensor = {
//    // Generate standard normal samples
//    val epsilon = Tensor.randn(sampleShape)
//    // Transform to our distribution parameters
//    mu.add(sigma.mul(epsilon))
//  }
}

// Companion object for convenient creation
object NormalDistribution {

  def apply[A <: Tuple](mu: Tensor[A], sigma: Tensor[A]): NormalDistribution[A] = {
    new NormalDistribution[A](mu, sigma)
  }
}