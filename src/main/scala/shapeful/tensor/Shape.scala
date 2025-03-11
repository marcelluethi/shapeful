package shapeful.tensor

import shapeful.tensor.Dimension
import scala.annotation.targetName


// type class  to access shape of a type
trait Shape[Dims <: Tuple] {
  def toList: List[Int]
}
object Shape {
  // Base case: empty tuple
  @targetName("emptytuple")
  given Shape[EmptyTuple] with {
    def toList: List[Int] = Nil
  }


  // Inductive case: head and tail
  given [H, T <: Tuple](using headDim: Dimension[H], tailShape: Shape[T]): Shape[H *: T] with {
    def toList: List[Int] = headDim.value :: tailShape.toList
  }



  // Helper method
  def apply[Dims <: Tuple](using shape: Shape[Dims]): Shape[Dims] = shape

  // Helper to create a shape from a list of integers (for runtime dimensions)
  def fromList(dims: List[Int]): List[Int] = dims
}
