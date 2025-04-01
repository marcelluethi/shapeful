package shapeful.tensor

case class TypedDim[A <: Singleton](val dim: Int)
extension [A <: Singleton](singleton: A)
  def ~>(dim: Int): TypedDim[A] = TypedDim[A](dim)

trait Shape
object Shape0 extends Shape
class Shape1[A <: Singleton](val dim1: Int) extends Shape
class Shape2[A <: Singleton, B <: Singleton](val dim1: Int, val dim2: Int)
    extends Shape
class Shape3[A <: Singleton, B <: Singleton, C <: Singleton](
    val dim1: Int,
    val dim2: Int,
    val dim3: Int
) extends Shape

object Shape:

  def apply[A <: Singleton](dim1: TypedDim[A]): Shape1[A] =
    new Shape1[A](dim1.dim)

  def apply[A <: Singleton, B <: Singleton](
      dim1: TypedDim[A],
      dim2: TypedDim[B]
  ): Shape2[A, B] =
    new Shape2[A, B](dim1.dim, dim2.dim)

  def apply[A <: Singleton, B <: Singleton, C <: Singleton](
      dim1: TypedDim[A],
      dim2: TypedDim[B],
      dim3: TypedDim[C]
  ): Shape3[A, B, C] =
    new Shape3[A, B, C](dim1.dim, dim2.dim, dim3.dim)
