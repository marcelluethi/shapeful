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
class Tensor[D <: Tuple](val stensor : torch.Tensor[Float32], val shape: List[Int]):
  override def toString(): String =
    stensor.toString()

  def grad(): Tensor[D] =
    val g = stensor.grad.get.detach()
    new Tensor(g, g.shape.toList)

  def copy(requiresGrad : Boolean): Tensor[D] =
    val cp = stensor.detach()
    cp.requiresGrad = requiresGrad
    new Tensor(cp, cp.shape.toList)


  inline def apply[A](index: Int) : Tensor[Remove[D, A]] =
    val dim = inlineIndexOf[D, A]
    val newTensor = torch.select(stensor, dim, index)

    new Tensor[Remove[D, A]](newTensor, newTensor.shape.toList)



  inline def sum[A]: Tensor[Remove[D, A]] =
    val i = inlineIndexOf[D, A]
    val newShape = shape.zipWithIndex.filter(_._2 != i).map(_._1)
    val newTensor = torch.sum(stensor, dim = i) //.sum(dim )
    new Tensor[Remove[D, A]](newTensor, newShape)

  inline def mean[A]: Tensor[Remove[D, A]] =
    val i = inlineIndexOf[D, A]
    val newShape = shape.zipWithIndex.filter(_._2 != i).map(_._1)
    val newt = torch.mean(stensor, dim = i)
    new Tensor[Remove[D, A]](newt, newt.shape.toList)


  inline def inlineIndexOf[T <: Tuple, A]: Int = inline erasedValue[T] match
    case _: (A *: _) => 0
    case _: (_ *: rest) => 1 + inlineIndexOf[rest, A]
    case _: EmptyTuple => -1

  // Helper type to remove an element from a tuple
  type Remove[T <: Tuple, A] <: Tuple = T match
    case A *: rest => rest
    case head *: rest => head *: Remove[rest, A]
    case EmptyTuple => EmptyTuple


extension [A, B](tensor: Tensor[(A, B)])
  def mult[C](other: Tensor[(B, C)]): Tensor[(A, C)] = {
    val newt = tensor.stensor.matmul(other.stensor)
    val t = new Tensor[(A, C)](newt, newt.shape.toList)
    t
  }

  def dot(other : Tensor[Tuple1[B]]) : Tensor[Tuple1[A]] =
    val newt = tensor.stensor `@` other.stensor
    new Tensor[(Tuple1[A])](newt, List(newt.shape.head))

  def cov : Tensor[(A, A)] =
    val n = tensor.shape.head
    val newt = tensor.stensor.matmul(tensor.stensor.t) * (1.0f / n)
    new Tensor[(A, A)](newt, List(newt.shape.head))

extension[A <: Tuple] (tensor: Tensor[A] )

  

  def multScalar(s : Tensor0) : Tensor[A] =
    val newt = tensor.stensor.mul(s.stensor)
    new Tensor[A](newt, newt.shape.toList)

  @targetName("add tensor")
  def add(other : Tensor[A]) : Tensor[A] = {
    val newt = tensor.stensor.add(other.stensor)
    new Tensor[A](newt, newt.shape.toList)
  }


  def div(other : Tensor[A]) : Tensor[A] =
    val newt = tensor.stensor.div(other.stensor)
    new Tensor[A](newt, newt.shape.toList)


  def divScalar(scalar : Tensor[Tuple1[1]]) : Tensor[A] =
    val newt = tensor.stensor.div(scalar.stensor)
    new Tensor[A](newt, newt.shape.toList)

  @targetName("add scalar")
  def addScalar(other: Tensor0): Tensor[A] = {
    val newt = tensor.stensor.add(other.stensor)
    new Tensor[A](newt, newt.shape.toList)
  }
  @targetName("subtract tensor")
  def sub(other: Tensor[A]): Tensor[A] = {
    val newt = tensor.stensor.sub(other.stensor)
    new Tensor[A](newt, newt.shape.toList)
  }

  def pow(exp : Int) : Tensor[A] =
    val newt = tensor.stensor.pow(exp)
    new Tensor[A] (newt, newt.shape.toList)

  def log: Tensor[A] =
    val newt = tensor.stensor.log
    new Tensor[A](newt, newt.shape.toList)




object Tensor {
  // Constructor with compile-time dimension checking
  inline def apply[D <: Tuple](initializer: => Float, requiresGrad : Boolean)(using shape: Shape[D]): Tensor[D] = {


    val size = shape.toList.product

    val data = Array.fill(size)(initializer)
    val sTensor =torch.Tensor(data).reshape(shape.toList *)
    sTensor.requiresGrad = requiresGrad
    new Tensor[D](sTensor, shape.toList)
  }

  inline def fromSeq[D <: Tuple](data : Seq[Float], requiresGrad: Boolean)(using shape: Shape[D]): Tensor[D] = {
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
