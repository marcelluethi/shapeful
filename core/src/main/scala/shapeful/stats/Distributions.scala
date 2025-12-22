package shapeful.stats

import shapeful.*
import shapeful.random.Random
import shapeful.jax.Jax
import shapeful.jax.Jax.scipy_stats as jstats
import shapeful.jax.Jax.PyDynamic

trait Distribution[T <: Tuple]:

  protected def jaxDist : Jax.PyDynamic 

  def logProb(x: Tensor[T]): Tensor0
  def sample(k : Random.Key): Tensor[T]



trait IndependentDistribution[T <: Tuple : Labels] extends Distribution[T]:

  def logProb(x: Tensor[T]): Tensor0 = 
    Tensor.fromPy(jaxDist.logpdf(x.jaxValue))

  def cdf(x: Tensor[T]): Tensor[T] = 
    Tensor.fromPy(jaxDist.cdf(x.jaxValue))

  def sample(k : Random.Key): Tensor[T]

  def asMultivariate: MultivariateDistribution[T] = new MultivariateDistribution[T]:
    
    override val jaxDist: Jax.PyDynamic = IndependentDistribution.this.jaxDist

    override def logProb(x: Tensor[T]) = IndependentDistribution.this.logProb(x)
    override def cdf(x: Tensor[T]) : Tensor0 = IndependentDistribution.this.cdf(x).sum
    override def sample(k: Random.Key) = IndependentDistribution.this.sample(k)



trait MultivariateDistribution[T <: Tuple] extends Distribution[T]:
  def logProb(x: Tensor[T]): Tensor0 = 
    Tensor.fromPy(jaxDist.logpdf(x.jaxValue))

  def cdf(x: Tensor[T]): Tensor0 = 
    Tensor.fromPy(jaxDist.cdf(x.jaxValue))
  def sample(k : Random.Key): Tensor[T]
