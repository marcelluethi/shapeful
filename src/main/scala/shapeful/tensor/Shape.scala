package shapeful.tensor

import scala.compiletime.{constValue, erasedValue, summonFrom}
import scala.Tuple.Concat


import shapeful.tensor.TupleHelpers.*
import scala.collection.immutable.IntMap

/**
 * Represents the shape of a tensor. Dims is a tuple with a type for each dimension.
 * Some operations assume that the types are unique, but this is not enforced.
 */
class Shape[Dims <: Tuple](dimTuple: ToIntTuple[Dims]):

    /**
     * Returns the dimensionality of the dimension with type A
     */
    inline def dim[A] : Int =
        val i : Int = inlineIndexOf[Dims, A]
        dimTuple(i).asInstanceOf[Int] 


    /**
     * returns a tuple consisting of the index of the dimension in the dimTuple together with the dimensionality of that dimension
     */ 
    inline def dimsWithIndexAux[T <: Tuple]: Tuple.Map[T, [X] =>> (Int, Int)] =
        inline erasedValue[T] match
        case _: EmptyTuple => EmptyTuple
        case _: (h *: t) => 
            val headDim =  dim[h] -> indexOf[h]
            val tailDims = dimsWithIndexAux[t]
            headDim *: tailDims


    def dims : List[Int] = dimTuple.toList.asInstanceOf[List[Int]]
    inline def dimsWithIndex[T <: Tuple] : List[(Int, Int)] = dimsWithIndexAux[T].toList.asInstanceOf[List[(Int, Int)]]

    def length : Int = dims.size

    inline def indexOf[A]: Int = inlineIndexOf[Dims, A]

    def ++ [OtherDimes <: Tuple](other : Shape[OtherDimes]): Shape[Concat[Dims, OtherDimes]] = 
        val newDimsList = dims ++ other.dims
        val newDims = Tuple.fromArray(newDimsList.toArray).asInstanceOf[ToIntTuple[Concat[Dims, OtherDimes]]]
        new Shape(newDims)

    inline def updateValue[A](dim : Int): Shape[Dims] = 
        val i = inlineIndexOf[Dims, A]
        val newDimList = dims.updated(i, dim)
        val newdims = Tuple.fromArray(newDimList.toArray).asInstanceOf[ToIntTuple[Dims]]
        new Shape[Dims](newdims)

    inline def removeKey[A]: Shape[Remove[Dims, A]] = 
        val i = inlineIndexOf[Dims, A]
        val dimlist : List[Int] = dimTuple.toList.asInstanceOf[List[Int]]
        val (front, rest) = dimlist.splitAt(i)
        val newDimList = front ++ rest.drop(1)
        val newdims = Tuple.fromArray(newDimList.toArray).asInstanceOf[ToIntTuple[Remove[Dims, A]]]
        new Shape(newdims)

    inline def removeKeys[T <: Tuple]: Shape[RemoveAll[Dims, T]] =
        inline erasedValue[T] match
            case _: EmptyTuple => 
                // If T is empty, return the current shape
                this.asInstanceOf[Shape[RemoveAll[Dims, T]]]
            case _: (h *: t) => 
                // Remove the head type first
                val intermediate = removeKey[h]
                // Then recursively remove the rest of the types
                intermediate.removeKeys[t].asInstanceOf[Shape[RemoveAll[Dims, T]]]

    inline def rename[NewDims <: Tuple](using ev: ValueOf[Tuple.Size[NewDims]], ev2: ValueOf[Tuple.Size[Dims]]): Shape[NewDims] = {
        inline if constValue[Tuple.Size[NewDims]] != constValue[Tuple.Size[Dims]] then
            compiletime.error("New dimensions must have the same length as original dimensions")
        new Shape[NewDims](dimTuple.asInstanceOf[ToIntTuple[NewDims]])
}

object Shape:
    def empty = new Shape[EmptyTuple.type](EmptyTuple) 
    def apply[A](n : Int) = new Shape[Tuple1[A]](Tuple1(n)) 
    def apply[A, B](m : Int, n : Int) = new Shape[(A, B)]((m, n))
    def apply[A, B, C](m : Int, n : Int, o : Int) = new Shape[(A, B, C)]((m, n, o))
    def apply[A, B, C, D](m : Int, n : Int, o : Int, p : Int) = new Shape[(A, B, C, D)]((m, n, o, p))
    def apply[A, B, C, D, E](m : Int, n : Int, o : Int, p : Int, q : Int) = new Shape[(A, B, C, D, E)]((m, n, o, p, q))

    

