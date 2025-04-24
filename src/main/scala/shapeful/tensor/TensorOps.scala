package shapeful.tensor

import torch.DType
import torch.Float32
import torch.DType.float32

/** Trait to create a tensor from a torch tensor */
trait FromRepr[DType <: torch.DType, T <: Tensor[DType]]:
  def createfromRepr(repr: torch.Tensor[DType]): T

/** operations that work on any tensor and simply transform each element without
  * changing the shape or type of the tensors
  */



object TensorOps:

  extension [DType <: torch.DType, T <: Tensor[DType]](t: T)
    def pow(p: Int)(using fromRepr: FromRepr[DType, T]): T =
      val tt: torch.Tensor[DType] = t.repr.pow(p).to(t.dtype)
      fromRepr.createfromRepr(tt)


    def log(using fromRepr: FromRepr[DType, T]): T =
      val tt: torch.Tensor[DType] = t.repr.log.to(t.dtype)
      fromRepr.createfromRepr(tt)

    def add(b: T | Tensor0[DType])(using fromRepr: FromRepr[DType, T]): T =
      val tt: torch.Tensor[DType] =
        t.repr.add(b.repr).asInstanceOf[torch.Tensor[DType]]
      fromRepr.createfromRepr(tt)

    def sub(b: T | Tensor0[DType])(using fromRepr: FromRepr[DType, T]): T =
      val tt: torch.Tensor[DType] =
        t.repr.sub(b.repr).asInstanceOf[torch.Tensor[DType]]
      fromRepr.createfromRepr(tt)

    def mul(b: T | Tensor0[DType])(using fromRepr: FromRepr[DType, T]): T =
      val tt: torch.Tensor[DType] =
        t.repr.mul(b.repr).asInstanceOf[torch.Tensor[DType]]
      fromRepr.createfromRepr(tt)

    def div(b: T | Tensor0[DType])(using fromRepr: FromRepr[DType, T]): T =
      val tt: torch.Tensor[DType] =
      t.repr.div(b.repr).asInstanceOf[torch.Tensor[DType]]
      fromRepr.createfromRepr(tt)

    def sqrt(using fromRepr: FromRepr[DType, T]): T =
      val tt: torch.Tensor[DType] = torch.sqrt(t.repr).to(t.dtype)
      fromRepr.createfromRepr(tt)
    
    def norm(using fromRepr: FromRepr[DType, T]): Tensor0[DType] =
      new Tensor0(torch.sum(t.mul(t).sqrt.repr), t.dtype)

    def abs(using fromRepr: FromRepr[DType, T]): T =
      val tt: torch.Tensor[DType] = t.repr.abs.to(t.dtype)
      fromRepr.createfromRepr(tt)