package shapeful.tensor

import scala.language.experimental.namedTypeArguments

import shapeful.*
import shapeful.jax.Jax
import me.shadaj.scalapy.py.SeqConverters

object TensorSlicing:

  // General splitAt for any tensor along any axis
  extension [T <: Tuple](t: Tensor[T])
    inline def splitAt[SplitAxis <: Label](index: Int): (Tensor[T], Tensor[T]) =
      val axisIndex = TupleHelpers.indexOf[SplitAxis, T]
      val axisSize = t.shape.dims(axisIndex)

      require(index >= 0 && index <= axisSize, s"Index $index out of bounds for axis size $axisSize")

      // Convert indices to JAX arrays
      val firstIndices = Jax.jnp.arange(0, index)
      val secondIndices = Jax.jnp.arange(index, axisSize)

      val first = Jax.jnp.take(t.jaxValue, firstIndices, axis = axisIndex)
      val second = Jax.jnp.take(t.jaxValue, secondIndices, axis = axisIndex)

      // Calculate new shapes
      val firstDims = t.shape.dims.updated(axisIndex, index)
      val secondDims = t.shape.dims.updated(axisIndex, axisSize - index)

      val firstShape = Shape[T](TupleHelpers.createTupleFromSeq[T](firstDims))
      val secondShape = Shape[T](TupleHelpers.createTupleFromSeq[T](secondDims))

      (new Tensor(firstShape, first, t.dtype), new Tensor(secondShape, second, t.dtype))
