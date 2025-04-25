package shapeful.tensor

import scala.compiletime.{erasedValue, summonInline}
import scala.collection.View.Single
import shapeful.Label

case class TypedDim[A <: Label](val dim: Int)
extension [A <: Label](Label: A)
  def ->>(dim: Int): TypedDim[A] = TypedDim[A](dim)



trait Shape:
  def dims : Seq[Int]

object Shape0 extends Shape:
  def dims: Seq[Int] = Seq()  
class Shape1[A <: Label](val dim1: Int) extends Shape:
  def dims = Seq(dim1)
class Shape2[A <: Label, B <: Label](val dim1: Int, val dim2: Int)
    extends Shape:
  def dims = Seq(dim1, dim2)
  inline def dim[D <: A | B]: Int = inline erasedValue[D] match {
    case _: A => dim1
    case _: B => dim2
    case _    => compiletime.error("Dimension must be either A or B")
  }


class Shape3[A <: Label, B <: Label, C <: Label](
    val dim1: Int,
    val dim2: Int,
    val dim3: Int
) extends Shape:
  def dims = Seq(dim1, dim2, dim3)
  inline def dim[D <: A | B | C]: Int = inline erasedValue[D] match {
    case _: A => dim1
    case _: B => dim2
    case _: C => dim3
    case _    => compiletime.error("Dimension must be either A or B")
  }


class Shape4[A <: Label, B <: Label, C <: Label, D <: Label](
    val dim1: Int,
    val dim2: Int,
    val dim3: Int,
    val dim4: Int
) extends Shape:
  def dims = Seq(dim1, dim2, dim3, dim4)
  inline def dim[Dim <: A | B | C | D ]: Int = inline erasedValue[Dim] match {
    case _: A => dim1
    case _: B => dim2
    case _: C => dim3
    case _: D => dim4
    case _    => compiletime.error("Dimension must be either A or B or C or D")
  }

object Shape:

  def apply[A <: Label](dim1: TypedDim[A]): Shape1[A] =
    new Shape1[A](dim1.dim)

  def apply[A <: Label, B <: Label](
      dim1: TypedDim[A],
      dim2: TypedDim[B]
  ): Shape2[A, B] =
    new Shape2[A, B](dim1.dim, dim2.dim)

  def apply[A <: Label, B <: Label, C <: Label](
      dim1: TypedDim[A],
      dim2: TypedDim[B],
      dim3: TypedDim[C]
  ): Shape3[A, B, C] =
    new Shape3[A, B, C](dim1.dim, dim2.dim, dim3.dim)

  def apply[A <: Label, B <: Label, C <: Label, D <: Label](
      dim1: TypedDim[A],
      dim2: TypedDim[B],
      dim3: TypedDim[C],
      dim4: TypedDim[D]  ): Shape4[A, B, C, D] =
    new Shape4[A, B, C, D](dim1.dim, dim2.dim, dim3.dim, dim4.dim)