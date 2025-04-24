package shapeful.distributions

import math.Pi
import shapeful.tensor.Tensor
import torch.Float32
import shapeful.tensor.Tensor0
import shapeful.tensor.TensorOps
import shapeful.tensor.FromRepr
import shapeful.tensor.TensorOps.pow
import shapeful.tensor.TensorOps.mul
import shapeful.tensor.TensorOps.sub
import shapeful.tensor.TensorOps.div
import shapeful.tensor.TensorOps.add
import shapeful.tensor.TensorOps.log

class Normal[T <: Tensor[Float32]](mu: T, sigma: T)(using
    fromRepr: FromRepr[Float32, T]
):

  // Precompute constants for efficiency
  private val variance = sigma.pow(2)
  private val logSigma = sigma.log
  private val log2Pi = Tensor0(math.log(2 * Pi).toFloat)

  /** Computes the log of the probability density function at x More numerically
    * stable for very small probabilities
    *
    * @param x
    *   Input tensor
    * @return
    *   log(PDF) values for each input
    */
  def logpdf(x: T)(using
      fromReprFloat32: FromRepr[Float32, Tensor0[Float32]]
  ): T =
    // Log PDF formula: -0.5 * log(2π) - log(σ) - 0.5 * ((x - μ)/σ)^2
    val term1 = Tensor0[Float32](-0.5f * math.log(2 * Pi).toFloat)
    val term2 = logSigma.mul(Tensor0[Float32](-0.5f))
    val xsubmu = x.sub(mu)
    val term3 = xsubmu.pow(2).div(variance).mul(Tensor0(-0.5))

    term2.add(term1).add(term3)

  /** Generate samples from the normal distribution
    *
    * @param sampleShape
    *   Shape of samples to generate
    * @return
    *   Samples from the normal distribution
    */
  def sample(): T = {
    val z = torch.randn(mu.repr.shape)
    val newtensor = mu.repr.add(sigma.repr.mul(z))
    fromRepr.createfromRepr(newtensor)
  }

