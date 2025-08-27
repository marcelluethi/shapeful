package shapeful.distributions

import scala.language.experimental.namedTypeArguments

import math.Pi
import shapeful.*

class MVNormal[L <: Label](mu: Tensor1[L], cov: Tensor2[L, L]):

  private val Lmat = Linalg.cholesky(cov) // Cholesky decomposition of the covariance matrix

  def shape: Shape1[L] = mu.shape

  def logpdf(x: Tensor1[L]): Tensor0 =
    val d = mu.shape.dim[L]
    val logdet = cov.det.log
    val prec = cov.inv
    val diff = x - mu
    val exponent = diff.dot(prec.matmul1(diff)) * Tensor0(-0.5f)
    val coeff = Tensor0((-0.5f) * d.toFloat * Math.log(2 * Math.PI).toFloat) - (logdet * Tensor0(0.5f))
    coeff + exponent

  def sample(key: shapeful.random.Random.Key): Tensor1[L] =
    val z = Normal(Tensor.zeros(mu.shape), Tensor.ones(mu.shape)).sample(key)
    Lmat.matmul1(z) + mu

  def sample[Sample <: Label](n: Int, key: shapeful.random.Random.Key): Tensor2[Sample, L] =
    val shape = Shape2[L, Sample](mu.shape.dim[L], n)
    val z = Normal(Tensor.zeros(shape), Tensor.ones(shape)).sample(key)
    val x = Lmat.matmul(z).transpose // L * z
    x.vmap[VmapAxis = Sample] { zSample => zSample + mu }

object MVNormal:

  def standardNormal[L <: Label](shape: Shape1[L]): MVNormal[L] =
    val mu = Tensor1[L](Seq.fill(shape.size)(0.0f))
    val cov = Tensor2.eye[L](Shape1(shape.size))
    new MVNormal(mu, cov)
