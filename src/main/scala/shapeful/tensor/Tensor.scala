package shapeful.tensor

import shapeful.tensor.Dimension.SymbolicTuple
import torch.Float32
import torch.indexing.---

import scala.annotation.targetName

import scala.compiletime.{constValue, erasedValue, summonFrom}
import scala.compiletime.ops.int.*
import scala.deriving.Mirror
import scala.reflect.ClassTag
import scala.compiletime.ops.boolean.*
import scala.compiletime.ops.any.*


import Tensor.Tensor0
import TupleHelpers.*
import torch.DType.float32
import torch.indexing.Slice

// Main tensor class - dimensions are encoded as type parameters
class Tensor[Dims <: Tuple](val shape : Shape[Dims], val stensor : torch.Tensor[Float32]):
  override def toString(): String =
    stensor.toString()


  def grad(): Tensor[Dims] =
    val g = stensor.grad.get.detach()
    new Tensor(shape, g)

  def copy(requiresGrad : Boolean): Tensor[Dims] =
    val cp = stensor.detach()
    cp.requiresGrad = requiresGrad
    new Tensor(shape, cp)


  def update(indices: ToIntTuple[Dims], value: Float): Unit =
    val idx = indices.productIterator.toList.map(_.asInstanceOf[Int]).toArray
    stensor.update(idx.toSeq, value)

  def get(indices: ToIntTuple[Dims]): Float = {
    val idx = indices.productIterator.toList.map(_.asInstanceOf[Int].toLong).toArray
    stensor(idx*).item
  }

  inline def split[SplitDim](n : Int): (Tensor[Dims], Tensor[Dims]) =
    val i = inlineIndexOf[Dims, SplitDim]
    val dims0 = (0 until shape.length).map(_ => ---).toList.updated(i, Slice(0, n))
    val stensor0 = stensor(dims0*)
    val dims1 = (0 until shape.length).map(_ => ---).toList.updated(i, Slice(n, shape.dim[SplitDim]))
    val stensor1 = stensor(dims1*)
    val newDims0 = shape.updateValue[SplitDim](n)
    val newDims1 = shape.updateValue[SplitDim](shape.dim[SplitDim] - n)

    val tensor0 = new Tensor(newDims0, stensor0)
    val tensor1 = new Tensor(newDims1, stensor1)
    (tensor0, tensor1)

  inline def dim[A] : Int = shape.dim[A]

  inline def apply[A](index: Int) : Tensor[Remove[Dims, A]] =
    val dim = inlineIndexOf[Dims, A]
    val newshape = shape.removeKey[A]
    val newTensor = torch.select(stensor, dim, index)

    new Tensor[Remove[Dims, A]](newshape, newTensor)


  inline def sum[A]: Tensor[Remove[Dims, A]] =
    val i = inlineIndexOf[Dims, A]
    val newShape = shape.removeKey[A]
    val newTensor = torch.sum(stensor, dim = i) //.sum(dim )
    new Tensor[Remove[Dims, A]](newShape, newTensor)

  inline def mean[A]: Tensor[Remove[Dims, A]] =
    val i = inlineIndexOf[Dims, A]
    val newShape = shape.removeKey[A]
    val newt = torch.mean(stensor, dim = i)
    new Tensor[Remove[Dims, A]](newShape, newt)

  inline def argmax[A] : Tensor[Remove[Dims, A]] =
    val i = inlineIndexOf[Dims, A]
    val newShape = shape.removeKey[A]
    val newt : torch.Tensor[Float32]= torch.argmax(stensor, dim = i).to(float32)
    new Tensor[Remove[Dims, A]](newShape, newt)
    




object Tensor {
  // Constructor with compile-time dimension checking
  inline def apply[Dims <: Tuple](shape : Shape[Dims], initializer: => Float, requiresGrad : Boolean=false): Tensor[Dims] = {


    val size = shape.dims.product

    val data = Array.fill(size)(initializer)
    val sTensor =torch.Tensor(data).reshape(shape.dims *)
    sTensor.requiresGrad = requiresGrad
    new Tensor[Dims](shape, sTensor)
  }

  inline def fromSeq[D <: Tuple](shape : Shape[D], data : Seq[Float], requiresGrad: Boolean=false): Tensor[D] = {
    val size = shape.dims.product
    require(data.size == size)


    val sTensor = torch.Tensor(data).reshape(shape.dims *)
    sTensor.requiresGrad = requiresGrad
    new Tensor[D](shape, sTensor)
  }


  type Tensor0 = Tensor[EmptyTuple.type] 

  type Tensor1[A] = Tensor[Tuple1[A]]
  type Tensor2[A, B] = Tensor[(A, B)]
  type Tensor3[A, B, C] = Tensor[(A, B, C)]
  type Tensor4[A, B, C, D] = Tensor[(A, B, C, D)]
}
