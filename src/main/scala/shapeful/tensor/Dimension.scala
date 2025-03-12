package shapeful.tensor

import scala.compiletime.constValue

trait Dimension[D] {
  def value: Int
}


object Dimension {
  // Define a simple case class implementation
  case class IntDimension[N <: Int & Singleton](value: Int) extends Dimension[N]
  
  // Use the case class in the given instance
  inline given dimensionInt[N <: Int & Singleton]: Dimension[N] = 
    IntDimension[N](constValue[N])


  // For symbolic dimensions (where we don't know the value at compile time)
  case class Symbolic[Label](override val value: Int) extends Dimension[Label]
  case class SymbolicTuple[Label]( _value: (Int, Int)) extends Dimension[Label] {
    override val value : Int = _value(0) * _value(1)
  }

  // Helper to create dimensions at runtime
  def apply[D](using d: Dimension[D]): Dimension[D] = d
}
