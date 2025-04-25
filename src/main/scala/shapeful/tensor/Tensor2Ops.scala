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
import javax.print.attribute.standard.MediaSize.Other
import shapeful.Label


object Tensor2Ops:

  type OtherDim[D <: Label, A <: Label, B <: Label] = D match {
    case A => B
    case B => A
  }


  extension [A <: Label, B <: Label, DType <: torch.DType](t: Tensor2[A, B, DType])
    def matmul[C <: Label](
        y: Tensor2[B, C, DType]
    ): Tensor2[A, C, DType] =
      new Tensor2[A, C, DType](new Shape2[A, C](t.shape.dim1, y.shape.dim2), t.repr.matmul(y.repr), t.dtype)

    
    def matmul1(y: Tensor1[B, DType]): Tensor1[A, DType] =
      new Tensor1[A, DType](new Shape1[A](t.shape.dim1), t.repr.matmul(y.repr), t.dtype)
    

    def inv: Tensor2[B, A, Float32] = 
      require(t.shape.dim1 == t.shape.dim2, "Tensor must be square to compute inverse")
      val floatT : Tensor2[A, B, Float32] = t.toType(float32)
      BasicLinalg.inverse(floatT, numIterations = 10, initialGuess = None)


    @targetName("addToRows")
    def addTensor1(
        y: Tensor1[B, DType]
    ): Tensor2[A, B, DType] =
      new Tensor2[A, B, DType](new Shape2[A, B](t.shape.dim1, t.shape.dim2), t.repr.add(y.repr), t.dtype)
   
    @targetName("addToCols")
    def addTensor1(
        y: Tensor1[A, DType]
    ): Tensor2[A, B, DType] = 
      new Tensor2[A, B, DType](new Shape2[A, B](t.shape.dim1, t.shape.dim2), t.repr.add(y.repr), t.dtype)

    // def getshape: (Int, Int) = (shape._1.n, shape._2.n)
    inline def sum[D <: A | B]: Tensor1[OtherDim[D, A, B], DType] =
      val dimInd = inline erasedValue[D] match {
        case _: A => 0
        case _: B => 1
        case _    => compiletime.error("Dimension must be either A or B")
      }
      val newt = torch.sum(t.repr, dimInd)
      val newShape = new Shape1[OtherDim[D, A, B]](newt.shape(0))
      new Tensor1(newShape, newt, t.dtype)

    def transpose: Tensor2[B, A, DType] =
      val newShape = new Shape2[B, A](t.shape.dim2, t.shape.dim1)
      new Tensor2[B, A, DType](newShape, t.repr.transpose(0, 1), t.dtype)

    inline def mean[D <: A | B]: Tensor1[OtherDim[D, A, B], Float32] =
      val dimInd = inline erasedValue[D] match {
              case _: A => 0
              case _: B => 1
              case _    => compiletime.error("Dimension must be either A or B")
            }

      val newt : torch.Tensor[Float32] = torch.mean(t.repr.to(float32), dim = dimInd).to(float32)
      new Tensor1[OtherDim[D, A, B], Float32](new Shape1(t.shape.dim1), newt, float32)


    inline def argmax[D <: A | B] : Tensor1[OtherDim[D, A, B], Int32] =

      val dimInd = inline erasedValue[D] match {
              case _: A => 0
              case _: B => 1
              case _    => compiletime.error("Dimension must be either A or B")
            }

      new Tensor1(new Shape1(t.shape.dim2), t.repr.argmax(dim = dimInd).to(int32), int32)
    
    inline def det: Tensor0[Float32] = 
      require(t.shape.dim1 == t.shape.dim2, "Tensor must be square to compute determinant")
      new Tensor0(torch.det(t.repr.to(float32)), float32)

    inline def map[D <: A | B](f : Tensor1[OtherDim[D, A, B], DType] => Tensor1[OtherDim[D, A, B], DType]): Tensor2[A, B, DType] =
      val dimInd = inline erasedValue[D] match {
              case _: A => 0
              case _: B => 1
              case _    => compiletime.error("Dimension must be either A or B")
            }
      
      val unbindRes = torch.unbind(t.repr, dimInd)
      val newRepr = torch.stack(unbindRes.tensorSeq.map(row =>      
        val rowt = new Tensor1[OtherDim[D, A, B], DType](new Shape1(row.shape(0)), row, row.dtype)
        f(rowt).repr
        ), dimInd)

    
      new Tensor2(t.shape, newRepr, newRepr.dtype)

    inline def reduce[D <: A | B](f : Tensor1[OtherDim[D, A, B], DType] => Tensor0[DType]): Tensor1[D, DType] =
      val dimInd = inline erasedValue[D] match {
              case _: A => 0
              case _: B => 1
              case _    => compiletime.error("Dimension must be either A or B")
            }
      
      val unbindRes = torch.unbind(t.repr, dimInd)
      val newRepr = torch.stack(unbindRes.tensorSeq.map(row =>      
        // val rowt = new Tensor1[OtherDim[D, A, B], DType](new Shape1(t.shape.dim[OtherDim[D, A, B]]), row, row.dtype)  
        // f(rowt).repr
        row.sum.to(t.dtype)
        ), 0).reshape(t.shape.dim[D])
      val newt = new Tensor1(new Shape1[D](t.shape.dim[D]), newRepr, newRepr.dtype)
      newt