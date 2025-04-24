package shapeful.tensor

import shapeful.tensor.Tensor0
import shapeful.tensor.Tensor1

object Tensor1Ops:

 extension [A <: Singleton, DType <: torch.DType](t: Tensor1[A, DType])
  def mean: Tensor0[DType] =
    new Tensor0[DType](t.repr.mean, t.dtype)

  def sum: Tensor0[DType] =
    new Tensor0[DType](torch.sum(t.repr), t.dtype)

  def dot(other: Tensor1[A, DType]): Tensor0[DType] =
    new Tensor0[DType](t.repr.dot(other.repr), t.dtype)

  
  def map[NewDType <: torch.DType](f: Tensor0[DType] => Tensor0[NewDType]): Tensor1[A, NewDType] =
    val newRepr = torch.stack(torch.unbind(t.repr, 0).tensorSeq.map(row =>      
        val rowt = new Tensor0[DType](row, row.dtype)
        f(rowt).repr
        ), 0)

    new Tensor1[A, NewDType](new Shape1[A](t.shape.dim1), newRepr, newRepr.dtype)

