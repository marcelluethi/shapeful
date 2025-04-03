package shapeful.tensor 

import shapeful.tensor.* 
import torch.Float32
import torch.DType.float32
import torch.Int32
import torch.DType.int32
import scala.compiletime.{erasedValue, summonInline}
import scala.annotation.targetName


object Tensor2Ops:

  type OtherDim[D <: Singleton, A <: Singleton, B <: Singleton] = D match {
    case A => B
    case B => A
  }


  extension [A <: Singleton, B <: Singleton, DType <: torch.DType](t: Tensor2[A, B, DType])
    def matmul[C <: Singleton](
        y: Tensor2[B, C, DType]
    ): Tensor2[A, C, DType] =
      new Tensor2[A, C, DType](new Shape2[A, C](t.shape.dim1, y.shape.dim2), t.repr.matmul(y.repr), t.dtype)

    
    def matmul1(y: Tensor1[B, DType]): Tensor1[A, DType] =
      new Tensor1[A, DType](new Shape1[A](t.shape.dim1), t.repr.matmul(y.repr), t.dtype)

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
      new Tensor2[B, A, DType](newShape, t.repr, t.dtype)

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
    