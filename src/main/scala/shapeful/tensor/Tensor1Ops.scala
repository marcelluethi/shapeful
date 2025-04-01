package shapeful.tensor

import shapeful.tensor.Tensor0
import shapeful.tensor.Tensor1

object Tensor1Ops:

 extension [A <: Singleton, DType <: torch.DType](t: Tensor1[A, DType])
  def mean: Tensor0[DType] =
    new Tensor0[DType](t.repr.mean, t.dtype)

  def sum: Tensor0[DType] =
    new Tensor0[DType](torch.sum(t.repr), t.dtype)

