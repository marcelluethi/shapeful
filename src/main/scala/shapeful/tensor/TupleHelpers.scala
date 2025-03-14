package shapeful.tensor


import scala.compiletime.{constValue, erasedValue, summonFrom}


object TupleHelpers {
    type Remove[T <: Tuple, A] <: Tuple = T match
        case A *: rest => rest
        case head *: rest => head *: Remove[rest, A]
        case EmptyTuple => EmptyTuple

    type ToIntTuple[T <: Tuple] = Tuple.Map[T, [X] =>> Int]

    inline def inlineIndexOf[T <: Tuple, A]: Int = inline erasedValue[T] match
        case _: (A *: _) => 0
        case _: (_ *: rest) => 1 + inlineIndexOf[rest, A]
        case _: EmptyTuple => -1

        
    type UpdateDim[Dims <: Tuple, Dim, NewDim] <: Tuple = Dims match {
        case Dim *: tail => NewDim *: tail
        case head *: tail => head *: UpdateDim[tail, Dim, NewDim]
        case EmptyTuple => EmptyTuple

    }
}
