package shapeful.tensor

import scala.compiletime.{error, erasedValue}

/** Contains basic helper functions for working with tuples (mostly on the type level)
  */
object TupleHelpers:

  /** Converts a tuple of any type to a tuple of Ints
    */
  type ToIntTuple[T <: Tuple] <: Tuple = T match
    case EmptyTuple => EmptyTuple
    case _ *: tail  => Int *: ToIntTuple[tail]

  /** Remove the first occurrence of an element A from a tuple B
    */
  type Remove[A, B <: Tuple] <: Tuple = B match
    case EmptyTuple => EmptyTuple
    case A *: EmptyTuple =>
      EmptyTuple // If removing the only element, return empty
    case A *: tail => tail // If removing first element, return tail
    case head *: tail =>
      head *: Remove[A, tail] // Otherwise, keep head and recurse

  /** Remove all occurrences of elements in ToRemove from From
    */
  type RemoveAll[ToRemove <: Tuple, From <: Tuple] <: Tuple = ToRemove match
    case EmptyTuple   => From
    case head *: tail => RemoveAll[tail, Remove[head, From]]

  // Successor type for counting (compact version)
  type S[N <: Int] <: Int = N match
    case 0  => 1
    case 1  => 2
    case 2  => 3
    case 3  => 4
    case 4  => 5
    case 5  => 6
    case 6  => 7
    case 7  => 8
    case 8  => 9
    case 9  => 10
    case 10 => 11
    case 11 => 12
    case 12 => 13
    case 13 => 14
    case 14 => 15

  // ========== Runtime operations ==========

  /** Helper method to create tuple from sequence (supports up to 6 elements)
    */
  def createTupleFromSeq[T <: Tuple](seq: Seq[Int]): ToIntTuple[T] =
    require(
      seq.length <= 6,
      s"Tuple size ${seq.length} not supported, maximum is 6"
    )

    seq.length match
      case 0 => EmptyTuple.asInstanceOf[ToIntTuple[T]]
      case 1 => Tuple1(seq(0)).asInstanceOf[ToIntTuple[T]]
      case 2 => (seq(0), seq(1)).asInstanceOf[ToIntTuple[T]]
      case 3 => (seq(0), seq(1), seq(2)).asInstanceOf[ToIntTuple[T]]
      case 4 => (seq(0), seq(1), seq(2), seq(3)).asInstanceOf[ToIntTuple[T]]
      case 5 =>
        (seq(0), seq(1), seq(2), seq(3), seq(4)).asInstanceOf[ToIntTuple[T]]
      case 6 =>
        (seq(0), seq(1), seq(2), seq(3), seq(4), seq(5))
          .asInstanceOf[ToIntTuple[T]]

  /** Get the index of the first occurrence of an element A in a tuple B
    */
  inline def indexOf[A, B <: Tuple]: Int =
    inline erasedValue[B] match
      case _: (A *: tail)    => 0
      case _: (head *: tail) => 1 + indexOf[A, tail]
      case _: EmptyTuple =>
        error("Element not found in tuple")

  /** Get indices of all elements from ToFind in InTuple
    */
  inline def indicesOf[ToFind <: Tuple, InTuple <: Tuple]: Tuple =
    inline erasedValue[ToFind] match
      case _: EmptyTuple => EmptyTuple
      case _: (head *: tail) =>
        indexOf[head, InTuple] *: indicesOf[tail, InTuple]
