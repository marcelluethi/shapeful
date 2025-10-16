package shapeful.distributions

import math.Pi
import shapeful.*
import shapeful.jax.Jax
import me.shadaj.scalapy.py.SeqConverters

/** Multivariate Normal distribution
  *
  * Models a joint probability distribution over d-dimensional vectors. Unlike Normal which treats each dimension
  * independently, MVNormal captures correlations between dimensions through the covariance matrix.
  *
  * Uses JAX's scipy.stats.multivariate_normal for efficient computation.
  *
  * @param mu
  *   Mean vector of shape [d]
  * @param cov
  *   Covariance matrix of shape [d, d] (must be positive definite)
  */
class MVNormal[L <: Label](val mu: Tensor1[L], val cov: Tensor2[L, L]) extends MultivariateDistribution[L]:

  /** Dimension of the distribution */
  def dim: Int = mu.shape.size

  /** Log probability density of a single vector
    *
    * Returns a scalar log probability for the input vector under the joint distribution.
    *
    * @param x
    *   A single d-dimensional vector
    * @return
    *   Scalar log probability
    */
  def logpdf(x: Tensor1[L]): Tensor0 =
    val logpdf_value = Jax.scipy_stats.multivariate_normal.logpdf(
      x.jaxValue,
      mean = mu.jaxValue,
      cov = cov.jaxValue
    )
    Tensor0(logpdf_value.as[Float])

  /** Log probability density for a batch of vectors
    *
    * Computes log probability independently for each vector in the batch.
    *
    * @param x
    *   Batch of vectors with shape [n, d]
    * @return
    *   Tensor1 of shape [n] with log probabilities
    */
  def logpdfBatch[Batch <: Label](x: Tensor2[Batch, L]): Tensor1[Batch] =
    val logpdf_values = Jax.scipy_stats.multivariate_normal.logpdf(
      x.jaxValue,
      mean = mu.jaxValue,
      cov = cov.jaxValue
    )
    new Tensor1[Batch](Shape1(x.shape.dim[Batch]), logpdf_values, mu.dtype)

  /** Sample a single vector from the distribution
    *
    * @param key
    *   Random key for sampling
    * @return
    *   d-dimensional sample vector
    */
  def sample(key: shapeful.random.Random.Key): Tensor1[L] =
    val samples = Jax.jrandom.multivariate_normal(
      key.jaxKey,
      mu.jaxValue,
      cov.jaxValue
    )
    new Tensor1[L](mu.shape, samples, mu.dtype)

  /** Sample n independent vectors from the distribution
    *
    * @param n
    *   Number of samples
    * @param key
    *   Random key for sampling
    * @return
    *   Batch of n samples with shape [n, d]
    */
  def sampleBatch[Batch <: Label](n: Int, key: shapeful.random.Random.Key): Tensor2[Batch, L] =
    val shape = Seq(n)
    val samples = Jax.jrandom.multivariate_normal(
      key.jaxKey,
      mu.jaxValue,
      cov.jaxValue,
      shape.toPythonProxy
    )
    val resultShape = Shape2[Batch, L](n, mu.shape.dim[L])
    new Tensor2[Batch, L](resultShape, samples, mu.dtype)

  /** Mean vector of the distribution */
  def mean: Tensor1[L] = mu

  /** Marginal variances (diagonal elements of covariance matrix) */
  def variance: Tensor1[L] =
    val diag = Jax.jnp.diag(cov.jaxValue)
    new Tensor1[L](mu.shape, diag, mu.dtype)

  /** Covariance matrix */
  def covariance: Tensor2[L, L] = cov

object MVNormal:

  def apply[L <: Label](mu: Tensor1[L], cov: Tensor2[L, L]): MVNormal[L] =
    new MVNormal(mu, cov)

  /** Standard normal distribution in d dimensions (zero mean, identity covariance) */
  def standard[L <: Label](shape: Shape1[L]): MVNormal[L] =
    val mu = Tensor1[L](Seq.fill(shape.size)(0.0f))
    val cov = Tensor2.eye[L](Shape1(shape.size))
    new MVNormal(mu, cov)
