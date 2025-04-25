package shapeful.distributions

import math.Pi
import shapeful.tensor.Tensor
import torch.Float32
import shapeful.tensor.{Tensor0, Tensor1, Tensor2, Tensor}
import shapeful.tensor.TensorOps
import shapeful.tensor.FromRepr
import shapeful.tensor.TensorOps.*
import shapeful.tensor.Tensor2
import shapeful.tensor.Tensor3
import shapeful.tensor.Tensor2Ops.*
import shapeful.tensor.Tensor1Ops.dot
import shapeful.linalg.BasicLinalg
import shapeful.tensor.Shape2
import shapeful.Label

class MVNormal[D <: Label](mu: Tensor1[D, Float32], cov: Tensor2[D, D, Float32]):

  private val L = BasicLinalg.cholesky(cov) // Cholesky decomposition of the covariance matrix

  def logpdf(x: Tensor1[D, Float32]): Tensor0[Float32] = {
    val d = mu.shape.dim1
    val logdet = cov.det.log
    val prec = cov.inv 
    val diff = x.sub(mu)
    val exponent = diff.dot(prec.matmul1(diff)).mul(Tensor0(-0.5f))
    val coeff = Tensor0((-0.5f) * d.toFloat * Math.log(2 * Math.PI).toFloat).sub(logdet.mul( Tensor0(0.5f)))
    coeff.add(exponent)
    
  }

  def sample(): Tensor1[D, Float32] = {
    val z = Normal(Tensor1(mu.shape, 0f), Tensor1(mu.shape, 1f)).sample()
    val x = L.matmul1(z) // L * z
    x.add(mu) // x = L * z + mu
  }

  def sample[S <: Label](n : Int): Tensor2[S, D, Float32] = {
    val shape = new Shape2[D, S](mu.shape.dim1, n)
    val z = Normal(Tensor2(shape, 0f), Tensor2(shape, 1f)).sample()
    val x = L.matmul(z).transpose // L * z
    x.addTensor1(mu) // x = L * z + mu
  }

  
