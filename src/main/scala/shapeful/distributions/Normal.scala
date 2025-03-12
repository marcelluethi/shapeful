package shapeful.distributions


import shapeful.tensor.Tensor
import shapeful.tensor.Tensor.{Tensor1, Tensor0}
import shapeful.tensor.TensorOps 

import scala.math.{Pi, exp, log, sqrt}
import shapeful.tensor.TensorOps.*

class NormalDistribution[Shape <: Tuple](mu: Tensor[Shape], sigma: Tensor[Shape]) {

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
  def logpdf(x: Tensor[Shape]): Tensor[Shape] = {
    // Log PDF formula: -0.5 * log(2π) - log(σ) - 0.5 * ((x - μ)/σ)^2
    val term1 = log2Pi.mul(Tensor(-0.5f, requiresGrad = false))
    val term2 = logSigma.mul(Tensor(-1f, requiresGrad = false))
    val term3 = x.sub(mu).pow(2).div(variance).mul(Tensor(-0.5f, requiresGrad = false))

    term2.add(term1).add(term3)
  }

  /**
   * Generate samples from the normal distribution
   *
   * @param sampleShape Shape of samples to generate
   * @return Samples from the normal distribution
   */
  def sample(): Tensor[Shape] = {
    val z = torch.randn(mu.stensor.shape)
    val newtensor = mu.stensor.add(sigma.stensor.mul(z))
    new Tensor(newtensor, newtensor.shape.toList)
  }
}

// Companion object for convenient creation
object NormalDistribution {

  def apply[Shape <: Tuple](mu: Tensor[Shape], sigma: Tensor[Shape]): NormalDistribution[Shape] = {
    new NormalDistribution[Shape](mu, sigma)
  }
}