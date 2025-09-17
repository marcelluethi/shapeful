package shapeful.tensor

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.jax.Jax
import shapeful.tensor.TupleHelpers
import me.shadaj.scalapy.py.SeqConverters

/** Advanced tensor indexing operations for efficient data manipulation.
  *
  * Provides:
  *   - slice: Extract contiguous ranges along an axis
  *   - gather: Select arbitrary indices along an axis
  */
object TensorIndexing:

  extension [T <: Tuple](tensor: Tensor[T])

    /** Extract a contiguous slice along the specified axis.
      *
      * @param start
      *   Starting index (inclusive)
      * @param end
      *   Ending index (exclusive)
      * @return
      *   New tensor with sliced data
      */
    inline def slice[SliceAxis <: Label](start: Int, end: Int): Tensor[T] =
      val axisIndex = TupleHelpers.indexOf[SliceAxis, T]
      val axisSize = tensor.shape.dims(axisIndex)

      require(start >= 0 && start <= axisSize, s"Start index $start out of bounds for axis size $axisSize")
      require(end >= start && end <= axisSize, s"End index $end invalid (start=$start, axisSize=$axisSize)")

      // Create indices for the slice range
      val indices = Jax.jnp.arange(start, end)

      // Use JAX take operation to slice along the specified axis
      val slicedJaxValue = Jax.jnp.take(tensor.jaxValue, indices, axis = axisIndex)

      // Calculate new shape
      val newDims = tensor.shape.dims.zipWithIndex.map { case (dim, idx) =>
        if idx == axisIndex then end - start else dim
      }
      val newShapeTuple = TupleHelpers.createTupleFromSeq[T](newDims)
      val newShape = Shape[T](newShapeTuple)

      new Tensor[T](newShape, slicedJaxValue, tensor.dtype)

    /** Gather elements along the specified axis using indices.
      *
      * @param indices
      *   Tensor containing indices to select
      * @return
      *   New tensor with gathered elements
      */
    inline def gather[GatherAxis <: Label, IndexLabel <: Label](indices: Tensor1[IndexLabel]): Tensor[T] =
      val axisIndex = TupleHelpers.indexOf[GatherAxis, T]
      val axisSize = tensor.shape.dims(axisIndex)

      // Convert indices to integers and validate bounds
      val indicesArray = Jax.jnp.astype(indices.jaxValue, Jax.jnp.int32)
      val minIndex = Jax.jnp.min(indicesArray).item().as[Int]
      val maxIndex = Jax.jnp.max(indicesArray).item().as[Int]

      require(
        minIndex >= 0 && maxIndex < axisSize,
        s"Indices out of bounds: [$minIndex, $maxIndex] for axis size $axisSize"
      )

      // Use JAX take operation to gather elements
      val gatheredJaxValue = Jax.jnp.take(tensor.jaxValue, indicesArray, axis = axisIndex)

      // Calculate new shape - the gather axis changes size to match indices
      val newDims = tensor.shape.dims.zipWithIndex.map { case (dim, idx) =>
        if idx == axisIndex then indices.shape.dims(0) else dim
      }
      val newShapeTuple = TupleHelpers.createTupleFromSeq[T](newDims)
      val newShape = Shape[T](newShapeTuple)

      new Tensor[T](newShape, gatheredJaxValue, tensor.dtype)

    /** Convenience method: slice with size instead of end index
      */
    inline def sliceWithSize[SliceAxis <: Label](start: Int, size: Int): Tensor[T] =
      slice[SliceAxis](start, start + size)

    /** Convenience method: gather using sequence of integers
      */
    inline def gatherSeq[GatherAxis <: Label](indices: Seq[Int]): Tensor[T] =
      val indicesTensor = Tensor1["indices"](indices.map(_.toFloat))
      gather[GatherAxis, "indices"](indicesTensor)

end TensorIndexing
