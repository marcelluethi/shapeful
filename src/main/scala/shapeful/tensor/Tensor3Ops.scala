package shapeful.tensor 

import shapeful.tensor.* 
import torch.Float32
import torch.DType.float32
import torch.Int32
import torch.DType.int32
import scala.compiletime.{erasedValue, summonInline}
import scala.annotation.targetName
import torch.Device.{CPU, CUDA}
import shapeful.linalg.BasicLinalg
import shapeful.Label


object Tensor3Ops:


  type Tensor2Selector[D <: Label, A <: Label, B <: Label, C <: Label, DType <: torch.DType] = D match {
    case A => Tensor2[B, C, DType]
    case B => Tensor2[A, C, DType]
    case C => Tensor2[A, B, DType]
  }


  extension [A <: Label, B <: Label, C <: Label, DType <: torch.DType](t: Tensor3[A, B, C, DType])

    inline def dimInd[D <: A | B | C]: Int = inline erasedValue[D] match {
      case _: A => 0
      case _: B => 1
      case _: C => 2
      case _    => compiletime.error("Dimension must be either A or B or C")
    }


    inline def reduce[D <: A | B | C](f : Tensor2Selector[D, A, B, C, DType] => Tensor0[DType])(using fromRepr : FromRepr[DType, Tensor2Selector[D, A, B, C, DType]]): Tensor1[D, DType] =
      
      val unbindRes = torch.unbind(t.repr, dimInd[D])
      val newRepr = torch.stack(unbindRes.tensorSeq.map(row =>      
        val rowt = fromRepr.createfromRepr(row)
        f(rowt).repr
        ), 0).reshape(t.shape.dim[D])

    
      new Tensor1(new Shape1[D](t.shape.dim[D]), newRepr, newRepr.dtype)
     
    inline def map[D <: A | B | C](f : Tensor2Selector[D, A, B, C, DType] => Tensor2Selector[D, A, B, C, DType])(using fromRepr : FromRepr[DType, Tensor2Selector[D, A, B, C, DType]]): Tensor3[A, B, C, DType] =
      val unbindRes = torch.unbind(t.repr, dimInd[D])
      val newRepr = torch.stack(unbindRes.tensorSeq.map(row =>      
        val d = dimInd[D]
        val rowt = fromRepr.createfromRepr(row)
        f(rowt).repr
        ), dimInd[D])
    
      new Tensor3(new Shape3[A, B, C](t.shape.dim1, t.shape.dim2, t.shape.dim3), newRepr, newRepr.dtype)
