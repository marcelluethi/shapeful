package shapeful.random

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.tensor.*
import shapeful.jax.{Jax, JaxDType}
import me.shadaj.scalapy.py.SeqConverters

/** JAX-based random number generation with proper key management.
  *
  * JAX uses a functional approach to randomness where:
  *   - Random keys must be explicitly managed
  *   - Keys are split to generate independent random streams
  *   - This ensures reproducibility and parallelizability
  *
  * This object provides low-level sampling primitives using JAX. For statistical modeling, prefer using distribution
  * classes in shapeful.distributions.
  */
object Random:

  /** A random key for generating random numbers */
  case class Key(jaxKey: Jax.PyDynamic):
    /** Split this key into multiple independent keys */
    def split(num: Int): Seq[Key] =
      val splitKeys = Jax.jrandom.split(jaxKey, num)
      (0 until num).map(i => Key(splitKeys.__getitem__(i)))

    /** Split into exactly 2 keys (common case) */
    def split2(): (Key, Key) =
      val keys = split(2)
      (keys(0), keys(1))

    /** Generate a new key by splitting */
    def next(): Key = split2()._2

  object Key:
    /** Create a random key from an integer seed */
    def apply(seed: Int): Key = Key(Jax.jrandom.key(seed))

    /** Create a random key from current time */
    def fromTime(): Key = Key(System.currentTimeMillis().toInt)

    /** Create a random key from Scala's random */
    def random(): Key = Key(scala.util.Random.nextInt())

  /** Generate random samples from various distributions.
    *
    * These are low-level JAX primitives. For statistical modeling with logpdf support, use the distribution classes in
    * shapeful.distributions instead.
    */

  /** Normal distribution with specified mean and standard deviation */
  def normal[T <: Tuple](
      key: Key,
      shape: Shape[T],
      mean: Tensor0 = Tensor0(0f),
      std: Tensor0 = Tensor0(1f),
      dtype: DType = DType.Float32
  ): Tensor[T] =
    val jaxValues = Jax.jrandom.normal(
      key.jaxKey,
      shape.dims.toPythonProxy,
      dtype = JaxDType.jaxDtype(dtype)
    )
    val standardNormal = new Tensor[T](shape, jaxValues, dtype)
    standardNormal * std + mean

  /** Uniform distribution in [0, 1) */
  def uniform[T <: Tuple](
      key: Key,
      shape: Shape[T],
      dtype: DType = DType.Float32
  ): Tensor[T] =
    val jaxValues = Jax.jrandom.uniform(
      key.jaxKey,
      shape.dims.toPythonProxy,
      dtype = JaxDType.jaxDtype(dtype)
    )
    new Tensor[T](shape, jaxValues, dtype)

  /** Uniform distribution in [minval, maxval) */
  def uniform[T <: Tuple](
      key: Key,
      shape: Shape[T],
      minval: Tensor0,
      maxval: Tensor0,
      dtype: DType
  ): Tensor[T] =
    val jaxValues = Jax.jrandom.uniform(
      key.jaxKey,
      shape.dims.toPythonProxy,
      minval = minval.jaxValue,
      maxval = maxval.jaxValue,
      dtype = JaxDType.jaxDtype(dtype)
    )
    new Tensor[T](shape, jaxValues, dtype)

  /** Bernoulli distribution (boolean outcomes) */
  def bernoulli[T <: Tuple](
      key: Key,
      shape: Shape[T],
      p: Tensor0
  ): Tensor[T] =
    val jaxValues = Jax.jrandom.bernoulli(
      key.jaxKey,
      p.jaxValue,
      shape.dims.toPythonProxy
    )
    new Tensor[T](shape, jaxValues, DType.Bool)

  /** Categorical distribution (sample integers from 0 to num_classes-1) */
  def categorical[T <: Tuple](
      key: Key,
      logits: Tensor[T],
      axis: Int = -1
  ): Tensor[T] =
    val jaxValues = Jax.jrandom.categorical(
      key.jaxKey,
      logits.jaxValue,
      axis = axis
    )
    new Tensor[T](logits.shape, jaxValues, DType.Int32)

  /** Sample from Gamma distribution */
  def gamma[T <: Tuple](
      key: Key,
      shape: Shape[T],
      alpha: Tensor0,
      beta: Tensor0 = Tensor0(1f),
      dtype: DType = DType.Float32
  ): Tensor[T] =
    val sample_value = Jax.jrandom.gamma(
      key.jaxKey,
      alpha.jaxValue,
      shape.dims.toPythonProxy,
      dtype = JaxDType.jaxDtype(dtype)
    )
    val scale = Tensor0(1.0f) / beta
    new Tensor[T](shape, sample_value, dtype) * scale

  /** Sample from Exponential distribution */
  def exponential[T <: Tuple](
      key: Key,
      shape: Shape[T],
      rate: Tensor0 = Tensor0(1f),
      dtype: DType = DType.Float32
  ): Tensor[T] =
    val exp_sample = Jax.jrandom.exponential(
      key.jaxKey,
      shape.dims.toPythonProxy,
      dtype = JaxDType.jaxDtype(dtype)
    )
    new Tensor[T](shape, exp_sample, dtype) / rate

  /** Sample from Cauchy distribution */
  def cauchy[T <: Tuple](
      key: Key,
      shape: Shape[T],
      loc: Tensor0 = Tensor0(0f),
      scale: Tensor0 = Tensor0(1f),
      dtype: DType = DType.Float32
  ): Tensor[T] =
    val cauchy_sample = Jax.jrandom.cauchy(
      key.jaxKey,
      shape.dims.toPythonProxy,
      dtype = JaxDType.jaxDtype(dtype)
    )
    new Tensor[T](shape, cauchy_sample, dtype) * scale + loc

  /** Sample from Beta distribution */
  def beta[T <: Tuple](
      key: Key,
      shape: Shape[T],
      alpha: Tensor0,
      beta: Tensor0,
      dtype: DType = DType.Float32
  ): Tensor[T] =
    val sample_value = Jax.jrandom.beta(
      key.jaxKey,
      a = alpha.jaxValue,
      b = beta.jaxValue,
      shape = shape.dims.toPythonProxy,
      dtype = JaxDType.jaxDtype(dtype)
    )
    new Tensor[T](shape, sample_value, dtype)

  /** Sample from Poisson distribution */
  def poisson[T <: Tuple](
      key: Key,
      shape: Shape[T],
      rate: Tensor0,
      dtype: DType = DType.Float32
  ): Tensor[T] =
    val sample_value = Jax.jrandom.poisson(
      key.jaxKey,
      lam = rate.jaxValue,
      shape = shape.dims.toPythonProxy,
      dtype = JaxDType.jaxDtype(dtype)
    )
    new Tensor[T](shape, sample_value, dtype)

  /** Sample from Binomial distribution */
  def binomial[T <: Tuple](
      key: Key,
      shape: Shape[T],
      n: Tensor0,
      p: Tensor0,
      dtype: DType = DType.Float32
  ): Tensor[T] =
    val samples = Jax.jrandom.binomial(
      key.jaxKey,
      n = n.jaxValue,
      p = p.jaxValue,
      shape = shape.dims.toPythonProxy,
      dtype = JaxDType.jaxDtype(dtype)
    )
    new Tensor[T](shape, samples, dtype)

  /** Randomly permute a tensor along an axis */
  private def permutation[T <: Tuple](
      key: Key,
      tensor: Tensor[T],
      axis: Int
  ): Tensor[T] =
    // JAX permutation requires at least 1-dimensional tensors
    if tensor.shape.rank == 0 then tensor // Return scalar tensors unchanged
    else
      val jaxValues = Jax.jrandom.permutation(
        key.jaxKey,
        tensor.jaxValue,
        axis = axis
      )
      new Tensor[T](tensor.shape, jaxValues, tensor.dtype)

  inline def permutation[T <: Tuple, PermutationAxis <: Label](
      key: Key,
      tensor: Tensor[T]
  ): Tensor[T] =
    val axisIndex = TupleHelpers.indexOf[PermutationAxis, T]
    permutation(key, tensor, axisIndex)

  /** Randomly sample indices without replacement */
  def choice[T <: Tuple](
      key: Key,
      shape: Shape[T],
      a: Int,
      replace: Boolean = true
  ): Tensor[T] =
    val jaxValues = Jax.jrandom.choice(
      key.jaxKey,
      a,
      shape.dims.toPythonProxy,
      replace = replace
    )
    new Tensor[T](shape, jaxValues, DType.Int32)

  // Utility functions for common patterns while staying functional

  /** Split a key into multiple keys and apply a function */
  def withSplitKeys[A](key: Key, numKeys: Int)(f: Seq[Key] => A): A =
    f(key.split(numKeys))

  /** Split a key into exactly 2 keys and apply a function */
  def withSplitKey2[A](key: Key)(f: (Key, Key) => A): A =
    val (k1, k2) = key.split2()
    f(k1, k2)

  /** Split a key into exactly 3 keys and apply a function */
  def withSplitKey3[A](key: Key)(f: (Key, Key, Key) => A): A =
    val keys = key.split(3)
    f(keys(0), keys(1), keys(2))

  /** Common pattern: generate multiple random tensors from one key */
  def multipleRandom[A](key: Key, count: Int)(f: Key => A): Seq[A] =
    key.split(count).map(f)

  /** Efficiently apply a function with different random keys using vmap for parallel execution */
  def vmapSample[R <: Tuple](
      baseKey: Key,
      count: Int,
      f: Key => Tensor[R]
  ): Tensor[Tuple.Concat[Tuple1["Sample"], R]] =
    import shapeful.tensor.{Tensor, Shape, TupleHelpers}

    // Split keys for parallel processing
    val keys = baseKey.split(count)

    // Use JAX's stack function to create an array of keys
    val keysArray = Jax.jnp.stack(keys.map(_.jaxKey).toPythonProxy)

    // Get sample result to determine output shape and dtype
    val sampleResult = f(keys.head)

    // Create vmapped function that uses JAX keys directly
    val fpy = (jaxKey: Jax.PyDynamic) =>
      val key = Key(jaxKey)
      f(key).jaxValue

    // Apply vmap for parallel execution
    val vmapped_fn = Jax.jax_helper.vmap(fpy, 0)
    val results = vmapped_fn(keysArray)

    // Create result tensor with proper shape: [Sample, ...originalShape]
    val resultDims = Seq(count) ++ sampleResult.shape.dims
    val resultTuple = TupleHelpers.createTupleFromSeq[Tuple.Concat[Tuple1["Sample"], R]](resultDims)
    val resultShape = Shape[Tuple.Concat[Tuple1["Sample"], R]](resultTuple)

    new Tensor(resultShape, results, sampleResult.dtype)
