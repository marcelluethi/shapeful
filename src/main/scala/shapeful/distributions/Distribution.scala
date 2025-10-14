package shapeful.distributions

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.random.Random

/** Base trait for all probability distributions
  * 
  * @tparam Support The type of values this distribution generates
  */
trait Distribution[Support]:
  /** Sample from the distribution */
  def sample(key: Random.Key): Support
  
  /** Log probability density/mass function */
  def logpdf(x: Support): Tensor0

/** Scalar distribution - represents a single random variable
  * 
  * Examples: Normal(0.0, 1.0), HalfNormal(1.0)
  */
trait ScalarDistribution extends Distribution[Tensor0]:
  def sample(key: Random.Key): Tensor0
  def logpdf(x: Tensor0): Tensor0
  
  /** Probability density/mass (less numerically stable) */
  def pdf(x: Tensor0): Tensor0 = logpdf(x).exp

/** Factorized distribution over independent elements
  * 
  * Represents a distribution over a tensor where each element is
  * an independent random variable with its own parameters.
  * This is the product distribution: p(x) = ∏ᵢ p(xᵢ)
  * 
  * Examples: Normal.factorized, Uniform.factorized
  */
trait FactorizedDistribution[S <: Tuple] extends Distribution[Tensor[S]]:
  /** The shape of samples from this distribution */
  def shape: Shape[S]
  
  /** Element-wise log probabilities (returns same shape as input) */
  def logpdfElements(x: Tensor[S]): Tensor[S]
  
  /** Joint log probability under independence assumption
    * 
    * Returns the sum of element-wise log probabilities,
    * exploiting the independence: log p(x) = Σᵢ log p(xᵢ)
    */
  final def logpdf(x: Tensor[S]): Tensor0 = logpdfElements(x).sum
  
  /** Sample from the distribution */
  def sample(key: Random.Key): Tensor[S]
  
  /** Probability density (less numerically stable) */
  def pdf(x: Tensor[S]): Tensor0 = logpdf(x).exp
  
  /** Mean of each element */
  def mean: Tensor[S]
  
  /** Variance of each element */
  def variance: Tensor[S]

/** Multivariate distribution over correlated dimensions
  * 
  * Represents a joint distribution over d-dimensional vectors
  * where dimensions may be correlated. Returns scalar log probability
  * for a single vector.
  * 
  * Examples: MVNormal, Dirichlet
  */
trait MultivariateDistribution[L <: Label] extends Distribution[Tensor1[L]]:
  /** Dimension of the distribution */
  def dim: Int
  
  /** Log probability density of a single vector (returns scalar) */
  def logpdf(x: Tensor1[L]): Tensor0
  
  /** Log probability density for a batch of vectors
    * 
    * @param x Batch of vectors with shape [n, d]
    * @return Tensor1 of shape [n] with log probabilities
    */
  def logpdfBatch[Batch <: Label](x: Tensor2[Batch, L]): Tensor1[Batch]
  
  /** Sample a single vector from the distribution */
  def sample(key: Random.Key): Tensor1[L]
  
  /** Sample n independent vectors from the distribution
    * 
    * @param n Number of samples
    * @param key Random key for sampling
    * @return Batch of n samples with shape [n, d]
    */
  def sampleBatch[Batch <: Label](n: Int, key: Random.Key): Tensor2[Batch, L]
  
  /** Probability density (less numerically stable than logpdf) */
  def pdf(x: Tensor1[L]): Tensor0 = logpdf(x).exp
  
  /** Mean vector of the distribution */
  def mean: Tensor1[L]
  
  /** Variance vector (marginal variances) */
  def variance: Tensor1[L]
  
  /** Covariance matrix (for distributions that have one) */
  def covariance: Tensor2[L, L]
