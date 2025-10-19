package shapeful.tensor

import TupleHelpers.{ToIntTuple, createTupleFromSeq}
import shapeful.Label
import scala.collection.View.Empty
import scala.annotation.publicInBinary

/** Represents the (typed) Shape of a tensor with runtime labels
  */
final class Shape[T <: Tuple] @publicInBinary private[tensor] (
    private val dimensions: ToIntTuple[T],
    private val labels: Array[String]
):

  def dims: Seq[Int] = dimensions.productIterator.toSeq.asInstanceOf[Seq[Int]]

  def axisLabels: Seq[String] = labels.toSeq

  inline def dim[D <: Label]: Int =
    val idx = TupleHelpers.indexOf[D, T]
    dimensions.productElement(idx).asInstanceOf[Int]

  inline def dim[D <: Label](axis: Axis[D]): Int =
    val idx = TupleHelpers.indexOf[D, T]
    dimensions.productElement(idx).asInstanceOf[Int]

  def relabel[NewT <: Tuple](using
      ev: Tuple.Size[T] =:= Tuple.Size[NewT]
  ): Shape[NewT] =
    new Shape[NewT](
      dimensions.asInstanceOf[ToIntTuple[NewT]],
      labels
    )

  def asTuple: ToIntTuple[T] = dimensions

  def *:[U <: Tuple](other: Shape[U]): Shape[Tuple.Concat[U, T]] =
    val leftDims = other.dims
    val rightDims = this.dims
    val combinedDims = leftDims ++ rightDims

    val combinedLabels = other.labels ++ this.labels

    val resultTuple = TupleHelpers.createTupleFromSeq[Tuple.Concat[U, T]](combinedDims)
    new Shape(resultTuple, combinedLabels)

  def size: Int = dims.product
  def rank: Int = dims.length

  override def toString: String =
    axisLabels.zip(dims).map((label, dim) => s"$label=$dim").mkString("Shape(", ", ", ")")

  override def equals(other: Any): Boolean = other match
    case s: Shape[?] => dimensions == s.dimensions && labels.sameElements(s.labels)
    case _           => false

  override def hashCode(): Int = dimensions.hashCode() ^ labels.toSeq.hashCode()

object Shape:

  def empty: Shape[EmptyTuple] = new Shape(EmptyTuple, Array.empty)

  def fromTuple[T <: Tuple](t: ToIntTuple[T]): Shape[T] =
    // Fallback without labels - use indices
    val labels = Array.tabulate(t.productArity)(i => s"dim$i")
    new Shape(t, labels)

  def apply[T <: Tuple](t: ToIntTuple[T]): Shape[T] = fromTuple(t)

  // Helper to extract label at compile time if available
  private inline def getLabel[L <: Label](idx: Int): String =
    scala.compiletime.summonFrom {
      case v: ValueOf[L] => v.value.toString
      case _             => s"dim$idx"
    }

  // Axis-based constructors with optional labels (using summonFrom to make ValueOf optional)
  inline def apply[L <: Label](dim: (Axis[L], Int)): Shape[L *: EmptyTuple] =
    val label = getLabel[L](0)
    new Shape(Tuple1(dim._2), Array(label))

  inline def apply[L1 <: Label, L2 <: Label](
      dim1: (Axis[L1], Int),
      dim2: (Axis[L2], Int)
  ): Shape[L1 *: L2 *: EmptyTuple] =
    val label1 = getLabel[L1](0)
    val label2 = getLabel[L2](1)
    new Shape((dim1._2, dim2._2), Array(label1, label2))

  inline def apply[L1 <: Label, L2 <: Label, L3 <: Label](
      dim1: (Axis[L1], Int),
      dim2: (Axis[L2], Int),
      dim3: (Axis[L3], Int)
  ): Shape[L1 *: L2 *: L3 *: EmptyTuple] =
    val label1 = getLabel[L1](0)
    val label2 = getLabel[L2](1)
    val label3 = getLabel[L3](2)
    new Shape(
      (dim1._2, dim2._2, dim3._2),
      Array(label1, label2, label3)
    )

  type Shape0 = Shape[EmptyTuple]
  type Shape1[L <: Label] = Shape[L *: EmptyTuple]
  type Shape2[L1 <: Label, L2 <: Label] = Shape[L1 *: L2 *: EmptyTuple]
  type Shape3[L1 <: Label, L2 <: Label, L3 <: Label] =
    Shape[L1 *: L2 *: L3 *: EmptyTuple]

val Shape0 = Shape.empty

object Shape1:
  def apply[L <: Label](dim: (Axis[L], Int))(using v: ValueOf[L]): Shape[L *: EmptyTuple] =
    Shape(dim)

object Shape2:
  def apply[L1 <: Label, L2 <: Label](
      dim1: (Axis[L1], Int),
      dim2: (Axis[L2], Int)
  )(using v1: ValueOf[L1], v2: ValueOf[L2]): Shape[L1 *: L2 *: EmptyTuple] =
    Shape(dim1, dim2)

object Test:
  val x = Shape2(Axis["A"] -> 3, Axis["B"] -> 4)

object Shape3:
  def apply[L1 <: Label, L2 <: Label, L3 <: Label](
      dim1: (Axis[L1], Int),
      dim2: (Axis[L2], Int),
      dim3: (Axis[L3], Int)
  )(using v1: ValueOf[L1], v2: ValueOf[L2], v3: ValueOf[L3]): Shape[L1 *: L2 *: L3 *: EmptyTuple] =
    Shape(dim1, dim2, dim3)
