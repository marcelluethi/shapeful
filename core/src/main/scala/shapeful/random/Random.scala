package shapeful.random

import shapeful.tensor.*
import shapeful.tensor.TensorOps.*
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

  class Normal[V](tv: TensorValue[V]):

    given TensorValue[V] = tv // TODO fix

    def apply[T <: Tuple: Labels](
        key: Key,
        shape: Shape[T],
        mean: Tensor0[V],
        std: Tensor0[V],
    ): Tensor[T, V] =
      val standardNormal = this(key, shape)
      standardNormal :* std :+ mean

    /** Normal distribution with mean=0 and std=1 */
    def apply[T <: Tuple: Labels](
        key: Key,
        shape: Shape[T],
    ): Tensor[T, V] = Tensor(tv).fromPy(
      Jax.jrandom.normal(
        key.jaxKey,
        shape.dimensions.toPythonProxy,
        dtype = tv.dtype.jaxType
      )
    )

  class Uniform[V](tv: TensorValue[V]):
    def apply[T <: Tuple: Labels](
        key: Key,
        shape: Shape[T],
        dtype: DType = DType.Float32
    ): Tensor[T, V] = 
      apply(key, shape, Tensor0(tv).zero, Tensor0(tv).one)

    /** Uniform distribution in [minval, maxval) */
    def apply[T <: Tuple: Labels](
        key: Key,
        shape: Shape[T],
        minval: Tensor0[V],
        maxval: Tensor0[V],
    ): Tensor[T, V] =
      val jaxValues = Jax.jrandom.uniform(
        key.jaxKey,
        shape.dimensions.toPythonProxy,
        minval = minval.jaxValue,
        maxval = maxval.jaxValue,
        dtype = tv.dtype.jaxType
      )
      Tensor(tv).fromPy(jaxValues)
