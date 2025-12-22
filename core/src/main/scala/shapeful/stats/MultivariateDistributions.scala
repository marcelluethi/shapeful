package shapeful.stats

import shapeful.*
import shapeful.random.Random
import shapeful.jax.Jax
import shapeful.jax.Jax.scipy_stats as jstats
import shapeful.jax.Jax.PyDynamic
import me.shadaj.scalapy.py.SeqConverters

class MVNormal[L : Label](
  val mean: Tensor1[L],
  val covariance: Tensor2[L, L],
) extends MultivariateDistribution[Tuple1[L]]:
  

  override val jaxDist : Jax.PyDynamic = jstats.multivariate_normal(
    mean = mean.jaxValue,
    cov = covariance.jaxValue,
  )

  override def sample(k: Random.Key): Tensor[Tuple1[L]] = 
    Random.multivariateNormal(k, mean, covariance, Shape(Axis[L] -> 1))


class Dirichlet[L : Label](
  val concentration: Tensor1[L],
) extends MultivariateDistribution[Tuple1[L]]:

  override val jaxDist: Jax.PyDynamic = jstats.dirichlet(
    alpha = concentration.jaxValue,
  )

  override def sample(k: Random.Key): Tensor1[L] = 
    Tensor.fromPy(Jax.jrandom.dirichlet(
      k.jaxKey, 
      alpha = concentration.jaxValue,
    ))

