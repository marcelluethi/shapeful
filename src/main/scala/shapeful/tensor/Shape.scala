package shapeful.tensor

import TupleHelpers.{ToIntTuple, createTupleFromSeq}
import shapeful.Label
import scala.collection.View.Empty

/** Represents the (typed) Shape of a tensor
  */
opaque type Shape[T <: Tuple] = ToIntTuple[T]

object Shape:

  def empty: Shape[EmptyTuple] = EmptyTuple

  def fromTuple[T <: Tuple](t: ToIntTuple[T]): Shape[T] = t

  // Generic constructor: create a Shape from an underlying tuple of Ints
  def apply[T <: Tuple](t: ToIntTuple[T]): Shape[T] = fromTuple(t)

  // Axis-based constructors (new ergonomic API)
  def apply[L <: Label](dim: (Axis[L], Int)): Shape[L *: EmptyTuple] =
    Tuple1(dim._2)

  def apply[L1 <: Label, L2 <: Label](dim1: (Axis[L1], Int), dim2: (Axis[L2], Int)): Shape[L1 *: L2 *: EmptyTuple] =
    (dim1._2, dim2._2)

  def apply[L1 <: Label, L2 <: Label, L3 <: Label](
      dim1: (Axis[L1], Int),
      dim2: (Axis[L2], Int),
      dim3: (Axis[L3], Int)
  ): Shape[L1 *: L2 *: L3 *: EmptyTuple] =
    (dim1._2, dim2._2, dim3._2)

  type Shape0 = Shape[EmptyTuple]
  type Shape1[L <: Label] = Shape[L *: EmptyTuple]
  type Shape2[L1 <: Label, L2 <: Label] = Shape[L1 *: L2 *: EmptyTuple]
  type Shape3[L1 <: Label, L2 <: Label, L3 <: Label] =
    Shape[L1 *: L2 *: L3 *: EmptyTuple]

  extension [T <: Tuple](s: Shape[T])
    def dims: Seq[Int] = s.productIterator.toSeq.asInstanceOf[Seq[Int]]

    // Two variants: (a) type-only: `shape.dim[Label]` and (b) runtime axis: `shape.dim(Axis[Label])`.
    inline def dim[D <: Label]: Int =
      val idx = TupleHelpers.indexOf[D, T]
      s.productElement(idx).asInstanceOf[Int]

    inline def dim[D <: Label](axis: Axis[D]): Int =
      // runtime axis parameter is ignored; index is computed from the type only
      val idx = TupleHelpers.indexOf[D, T]
      s.productElement(idx).asInstanceOf[Int]

    def relabel[NewT <: Tuple](using
        ev: Tuple.Size[T] =:= Tuple.Size[NewT]
    ): Shape[NewT] =
      Shape.fromTuple(s.asInstanceOf[ToIntTuple[NewT]])

    def asTuple: ToIntTuple[T] = s

    /** Concatenate shapes to form a larger product type. This creates a new shape by joining dimensions, similar to
      * tuple concatenation with *: Example: Shape(3, 4) *: Shape(5, 6) = Shape(3, 4, 5, 6)
      */
    def *:[U <: Tuple](other: Shape[U]): Shape[Tuple.Concat[T, U]] =
      val leftDims = s.dims
      val rightDims = other.dims
      val combinedDims = leftDims ++ rightDims

      val resultTuple =
        TupleHelpers.createTupleFromSeq[Tuple.Concat[T, U]](combinedDims)
      Shape.fromTuple(resultTuple)

    // Add common tensor operations
    def size: Int = dims.product
    def rank: Int = dims.length

val Shape0 = Shape.empty

object Shape1:
  def apply[L <: Label](dim: (Axis[L], Int)): Shape[L *: EmptyTuple] =
    Shape(dim)

object Shape2:
  def apply[L1 <: Label, L2 <: Label](dim1: (Axis[L1], Int), dim2: (Axis[L2], Int)): Shape[L1 *: L2 *: EmptyTuple] =
    Shape(dim1, dim2)

object Test:
  val x = Shape2(Axis["A"] -> 3, Axis["B"] -> 4)

object Shape3:
  def apply[L1 <: Label, L2 <: Label, L3 <: Label](
      dim1: (Axis[L1], Int),
      dim2: (Axis[L2], Int),
      dim3: (Axis[L3], Int)
  ): Shape[L1 *: L2 *: L3 *: EmptyTuple] =
    Shape(dim1, dim2, dim3)
