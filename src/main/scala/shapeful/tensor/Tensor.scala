package shapeful.tensor

import shapeful.tensor.Dimension.SymbolicTuple
import torch.Float32

import scala.annotation.targetName

import scala.compiletime.{constValue, erasedValue, summonFrom}
import scala.compiletime.ops.int.*
import scala.deriving.Mirror
import scala.reflect.ClassTag
import scala.compiletime.ops.boolean.*
import scala.compiletime.ops.any.*

import Tensor.Tensor0

// Main tensor class - dimensions are encoded as type parameters
class Tensor[Dims <: Tuple](val stensor : torch.Tensor[Float32], val shape: List[Int]):
  override def toString(): String =
    stensor.toString()

  def grad(): Tensor[Dims] =
    val g = stensor.grad.get.detach()
    new Tensor(g, g.shape.toList)

  def copy(requiresGrad : Boolean): Tensor[Dims] =
    val cp = stensor.detach()
    cp.requiresGrad = requiresGrad
    new Tensor(cp, cp.shape.toList)


  def update(indices: Tuple.Map[Dims, [X] =>> Int], value: Float): Unit =
    val idx = indices.productIterator.toList.map(_.asInstanceOf[Int]).toArray
    stensor.update(idx.toSeq, value)

  def get(indices: Tuple.Map[Dims, [X] =>> Int]): Float = {
    val idx = indices.productIterator.toList.map(_.asInstanceOf[Int].toLong).toArray
    stensor(idx: _*).item
  }

  inline def apply[A](index: Int) : Tensor[Remove[Dims, A]] =
    val dim = inlineIndexOf[Dims, A]
    val newTensor = torch.select(stensor, dim, index)

    new Tensor[Remove[Dims, A]](newTensor, newTensor.shape.toList)


  inline def sum[A]: Tensor[Remove[Dims, A]] =
    val i = inlineIndexOf[Dims, A]
    val newShape = shape.zipWithIndex.filter(_._2 != i).map(_._1)
    val newTensor = torch.sum(stensor, dim = i) //.sum(dim )
    new Tensor[Remove[Dims, A]](newTensor, newShape)

  inline def mean[A]: Tensor[Remove[Dims, A]] =
    val i = inlineIndexOf[Dims, A]
    val newShape = shape.zipWithIndex.filter(_._2 != i).map(_._1)
    val newt = torch.mean(stensor, dim = i)
    new Tensor[Remove[Dims, A]](newt, newt.shape.toList)


  inline def inlineIndexOf[T <: Tuple, A]: Int = inline erasedValue[T] match
    case _: (A *: _) => 0
    case _: (_ *: rest) => 1 + inlineIndexOf[rest, A]
    case _: EmptyTuple => -1

  // Helper type to remove an element from a tuple
  type Remove[T <: Tuple, A] <: Tuple = T match
    case A *: rest => rest
    case head *: rest => head *: Remove[rest, A]
    case EmptyTuple => EmptyTuple







object Tensor {
  // Constructor with compile-time dimension checking
  inline def apply[Dims <: Tuple](initializer: => Float, requiresGrad : Boolean=false)(using shape: Shape[Dims]): Tensor[Dims] = {


    val size = shape.toList.product

    val data = Array.fill(size)(initializer)
    val sTensor =torch.Tensor(data).reshape(shape.toList *)
    sTensor.requiresGrad = requiresGrad
    new Tensor[Dims](sTensor, shape.toList)
  }

  inline def fromSeq[D <: Tuple](data : Seq[Float], requiresGrad: Boolean=false)(using shape: Shape[D]): Tensor[D] = {
    val size = shape.toList.product
    require(data.size == size)


    val sTensor = torch.Tensor(data).reshape(shape.toList *)
    sTensor.requiresGrad = requiresGrad
    new Tensor[D](sTensor, shape.toList)
  }


  type Tensor0 = Tensor[EmptyTuple.type] 

  type Tensor1[A] = Tensor[Tuple1[A]]
  type Tensor2[A, B] = Tensor[(A, B)]
  type Tensor3[A, B, C] = Tensor[(A, B, C)]
  type Tensor4[A, B, C, D] = Tensor[(A, B, C, D)]
}
