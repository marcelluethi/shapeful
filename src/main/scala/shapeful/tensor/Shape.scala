package shapeful.tensor

import TupleHelpers.{ToIntTuple, createTupleFromSeq}
import shapeful.Label

/** Represents the (typed) Shape of a tensor
  */
opaque type Shape[T <: Tuple] = ToIntTuple[T]

object Shape:

  def apply[T <: Tuple](t: ToIntTuple[T]): Shape[T] = t
  def empty: Shape[EmptyTuple] = Shape(EmptyTuple)

  type Shape0 = Shape[EmptyTuple]
  type Shape1[L <: Label] = Shape[L *: EmptyTuple]
  type Shape2[L1 <: Label, L2 <: Label] = Shape[L1 *: L2 *: EmptyTuple]
  type Shape3[L1 <: Label, L2 <: Label, L3 <: Label] =
    Shape[L1 *: L2 *: L3 *: EmptyTuple]

  extension [T <: Tuple](s: Shape[T])
    def dims: Seq[Int] = s.productIterator.toSeq.asInstanceOf[Seq[Int]]

    inline def dim[D <: Label]: Int =
      val idx = TupleHelpers.indexOf[D, T]
      s.productElement(idx).asInstanceOf[Int]

    def relabel[NewT <: Tuple](using
        ev: Tuple.Size[T] =:= Tuple.Size[NewT]
    ): Shape[NewT] =
      Shape(s.asInstanceOf[ToIntTuple[NewT]])

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
      Shape(resultTuple)

    // Add common tensor operations
    def size: Int = dims.product
    def rank: Int = dims.length

val Shape0 = Shape.empty

object Shape1:
  def apply[L <: Label](dim: Int): Shape[L *: EmptyTuple] =
    Shape(Tuple1(dim))

object Shape2:
  def apply[L1 <: Label, L2 <: Label](
      dim1: Int,
      dim2: Int
  ): Shape[L1 *: L2 *: EmptyTuple] =
    Shape((dim1, dim2))

object Shape3:
  def apply[L1 <: Label, L2 <: Label, L3 <: Label](
      dim1: Int,
      dim2: Int,
      dim3: Int
  ): Shape[L1 *: L2 *: L3 *: EmptyTuple] =
    Shape((dim1, dim2, dim3))
