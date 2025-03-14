package shapeful.distributions


import shapeful.tensor.Tensor
import shapeful.tensor.Tensor.{Tensor1, Tensor0}
import shapeful.tensor.TensorOps 

import scala.math.{Pi, exp, log, sqrt}
import shapeful.tensor.TensorOps.*
import shapeful.tensor.Shape

class Normal[Dims <: Tuple](shape : Shape[Dims], mu: Tensor[Dims], sigma: Tensor[Dims]) {

  // Precompute constants for efficiency
  private val variance = sigma.pow(2)
  private val logSigma = sigma.log
  private val log2Pi : Tensor0 = Tensor(Shape.empty, math.log(2 * Pi).toFloat, requiresGrad = false)


  /**
   * Computes the log of the probability density function at x
   * More numerically stable for very small probabilities
   *
   * @param x Input tensor
   * @return log(PDF) values for each input
   */
  def logpdf(x: Tensor[Dims]): Tensor[Dims] = {
    // Log PDF formula: -0.5 * log(2π) - log(σ) - 0.5 * ((x - μ)/σ)^2
    val term1 = log2Pi.mul(Tensor(Shape.empty, -0.5f, requiresGrad = false))
    val term2 = logSigma.mul(Tensor(Shape.empty, -1f, requiresGrad = false))
    val term3 = x.sub(mu).pow(2).div(variance).mul(Tensor(Shape.empty, -0.5f, requiresGrad = false))

    term2.add(term1).add(term3)
  }

  /**
   * Generate samples from the normal distribution
   *
   * @param sampleShape Shape of samples to generate
   * @return Samples from the normal distribution
   */
  def sample(): Tensor[Dims] = {
    val z = torch.randn(mu.stensor.shape)
    val newtensor = mu.stensor.add(sigma.stensor.mul(z))
    new Tensor(shape, newtensor)
  }
}

// Companion object for convenient creation
object Normal {

  def apply[Dims <: Tuple](shape : Shape[Dims], mu: Tensor[Dims], sigma: Tensor[Dims]): Normal[Dims] = {
    new Normal[Dims](shape, mu, sigma)
  }
}