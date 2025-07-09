package shapeful.distributions

import scala.language.experimental.namedTypeArguments

import math.Pi
import shapeful.*

class MVNormal[L <: Label](mu: Tensor1[L], cov: Tensor2[L, L]):

  private val Lmat = Linalg.cholesky(cov) // Cholesky decomposition of the covariance matrix

  def logpdf(x: Tensor1[L]): Tensor0 = {
    val d = mu.shape.dim[L]
    val logdet = cov.det.log
    val prec = cov.inv 
    val diff = x - mu
    val exponent = diff.dot(prec.matmul1(diff)) * Tensor0(-0.5f)
    val coeff = Tensor0((-0.5f) * d.toFloat * Math.log(2 * Math.PI).toFloat) - (logdet * Tensor0(0.5f))
    coeff +  exponent
    
  }

  def sample(): Tensor1[L] = {
    val z = Normal(Tensor.zeros(mu.shape), Tensor.ones(mu.shape)).sample()
    Lmat.matmul1(z) + mu 
  }

  def sample[Sample <: Label](n : Int): Tensor2[Sample, L] = {
    val shape = Shape2[L, Sample](mu.shape.dim[L], n)
    val z = Normal(Tensor.zeros(shape), Tensor.ones(shape)).sample()
    val x = Lmat.matmul(z).transpose // L * z
    x.vmap[VmapAxis=Sample] { z =>  z + mu  }
  }
