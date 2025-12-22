package shapeful.stats

import shapeful.*
import shapeful.random.Random
import shapeful.jax.Jax
import shapeful.jax.Jax.scipy_stats as jstats
import shapeful.jax.Jax.PyDynamic
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters



class IndependentNormal[T <: Tuple : Labels](
  val mean: Tensor[T],
  val stddev: Tensor[T],
) extends  IndependentDistribution[T]:
  require(mean.shape == stddev.shape, "Mean and stddev must have the same shape")

  override protected def jaxDist: PyDynamic = jstats.norm(
    loc = mean.jaxValue,
    scale = stddev.jaxValue,
  )

  override def logProb(x: Tensor[T]): Tensor0 = 
    Tensor.fromPy(jaxDist.logpdf(x.jaxValue))
  override def cdf(x: Tensor[T]): Tensor[T] = 
    Tensor.fromPy(jaxDist.cdf(x.jaxValue))
  override def sample(k: Random.Key): Tensor[T] = 
    Random.normal(k, mean.shape)


class Uniform[T <: Tuple : Labels](
  val low: Tensor[T],
  val high: Tensor[T],
) extends IndependentDistribution[T]:
  require(low.shape == high.shape, "Low and high must have the same shape")

  override protected def jaxDist: PyDynamic = jstats.uniform(
    loc = low.jaxValue,
    scale = (high - low).jaxValue,
  )

  override def sample(k: Random.Key): Tensor[T] = 
    val uniform01 = Random.uniform(k, low.shape)
    uniform01 :* (high - low) :+ low


class Bernoulli[T <: Tuple : Labels](
  val probs: Tensor[T],
) extends IndependentDistribution[T]:

  override protected def jaxDist: PyDynamic = jstats.bernoulli(
    p = probs.jaxValue,
  )

  override def sample(k: Random.Key): Tensor[T] = 
    Tensor.fromPy(Jax.jrandom.bernoulli(k.jaxKey, p = probs.jaxValue)).asType(DType.Int32)


class Multinomial[L : Label](
  val n: Int,
  val probs: Tensor1[L],
) extends IndependentDistribution[Tuple1[L]]:

  private lazy val logProbs: Tensor1[L] = probs.log

  override protected def jaxDist: PyDynamic = jstats.multinomial(
    n = n,
    p = probs.jaxValue,
  )

  override def sample(k: Random.Key): Tensor1[L] = 
    Tensor.fromPy[Tuple1[L]](Jax.jrandom.multinomial(
      k.jaxKey,
      n = n,
      pvals = probs.jaxValue,
    )).asType(DType.Int32)


class Categorical[L : Label](
  val probs: Tensor1[L],
) extends IndependentDistribution[EmptyTuple]:

  // Categorical is Multinomial with n=1, returning the sampled index instead of counts
  private val multinomial = Multinomial(1, probs)
  private lazy val logProbs: Tensor1[L] = probs.log

  override protected def jaxDist: PyDynamic = 
    py.Dynamic.global.selectDynamic("None")

  override def logProb(x: Tensor0): Tensor0 = 
    val idx = x.toInt
    val logProbAtIdx = Tensor0(logProbs.jaxValue.bracketAccess(idx).as[Jax.PyDynamic])
    // Normalize: log(p_i) - log(sum_j p_j) = log(p_i) since probs sum to 1
    logProbAtIdx

  override def cdf(x: Tensor0): Tensor0 = 
    throw new UnsupportedOperationException("CDF not defined for Categorical distribution")

  override def sample(k: Random.Key): Tensor0 = 
    Tensor0(Jax.jrandom.categorical(k.jaxKey, logits = logProbs.jaxValue)).asType(DType.Int32)


class Cauchy[T <: Tuple : Labels](
  val loc: Tensor[T],
  val scale: Tensor[T],
) extends IndependentDistribution[T]:
  require(loc.shape == scale.shape, "Location and scale must have the same shape")

  override protected def jaxDist: PyDynamic = jstats.cauchy(
    loc = loc.jaxValue,
    scale = scale.jaxValue,
  )

  override def sample(k: Random.Key): Tensor[T] = 
    Tensor.fromPy(Jax.jrandom.cauchy(k.jaxKey, shape = loc.shape.dimensions.toPythonProxy)) * scale + loc


class HalfNormal[T <: Tuple : Labels](
  val scale: Tensor[T],
) extends IndependentDistribution[T]:

  override protected def jaxDist: PyDynamic = jstats.halfnorm(
    loc = Tensor0(0f).jaxValue,
    scale = scale.jaxValue,
  )

  override def sample(k: Random.Key): Tensor[T] = 
    Random.normal(k, scale.shape).abs * scale


class StudentT[T <: Tuple : Labels](
  val df: Int,
  val loc: Tensor[T],
  val scale: Tensor[T],
) extends IndependentDistribution[T]:
  require(loc.shape == scale.shape, "loc, and scale must have the same shape")

  override protected def jaxDist: PyDynamic = jstats.t(
    df = df,
    loc = loc.jaxValue,
    scale = scale.jaxValue,
  )

  override def sample(k: Random.Key): Tensor[T] = 
    Tensor.fromPy(Jax.jrandom.t(k.jaxKey, df = df, shape = loc.shape.dimensions.toPythonProxy)) * scale + loc

