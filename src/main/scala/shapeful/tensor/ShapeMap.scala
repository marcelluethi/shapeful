package shapeful.tensor

import scala.reflect.ClassTag
import scala.compiletime.{constValue, erasedValue, summonFrom}
import scala.annotation.targetName

// Type class to represent dimensions at runtime
trait DimensionRepresentation[A] {
  def key: ClassTag[A]
}

object DimensionRepresentation {
  given intDimensionRep[N <: Int & Singleton](using ct: ClassTag[N]): DimensionRepresentation[N] = new DimensionRepresentation[N] {
    def key: ClassTag[N] = ct
  }

  given symbolicDimensionRep[Label](using ct: ClassTag[Label]): DimensionRepresentation[Label] = new DimensionRepresentation[Label] {
    def key: ClassTag[Label] = ct
  }
}

// type class  to access shape (the value) from a tuple and provide dimension keys
trait ShapeM[Dims <: Tuple] {
  def toList: List[Int]
  def dimensionKeys: List[ClassTag[?]]
}

object ShapeM {
  // Base case: empty tuple
  @targetName("emptytuple")
  given ShapeM[EmptyTuple] with {
    def toList: List[Int] = Nil
    def dimensionKeys: List[ClassTag[?]] = Nil
  }

  // Inductive case: head and tail
  given [H, T <: Tuple](using headDim: Dimension[H], tailShape: ShapeM[T], dimRep: DimensionRepresentation[H]): ShapeM[H *: T] with {
    def toList: List[Int] = Dimension(using headDim).value :: tailShape.toList
    def dimensionKeys: List[ClassTag[?]] = dimRep.key :: tailShape.dimensionKeys
  }

  // Helper method
  def apply[Dims <: Tuple](using shape: ShapeM[Dims]): ShapeM[Dims] = shape
}