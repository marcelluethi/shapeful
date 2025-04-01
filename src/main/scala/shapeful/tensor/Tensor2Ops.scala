package shapeful.tensor 

import shapeful.tensor.* 
import torch.Float32
import torch.DType.float32
import torch.Int32
import torch.DType.int32


object Tensor2Ops:
  extension [A <: Singleton, B <: Singleton, DType <: torch.DType](t: Tensor2[A, B, DType])
    def matmul[C <: Singleton](
        y: Tensor2[B, C, DType]
    ): Tensor2[A, C, DType] =
      new Tensor2[A, C, DType](new Shape2[A, C](t.shape.dim1, y.shape.dim2), t.repr.matmul(y.repr), t.dtype)

    
    def matmul1(y: Tensor1[B, DType]): Tensor1[A, DType] =
      new Tensor1[A, DType](new Shape1[A](t.shape.dim1), t.repr.matmul(y.repr), t.dtype)

    def addToCols(
        y: Tensor1[B, DType]
    ): Tensor2[A, B, DType] =
      new Tensor2[A, B, DType](new Shape2[A, B](t.shape.dim1, t.shape.dim2), t.repr.add(y.repr), t.dtype)
    
    def addToRows(
        y: Tensor1[A, DType]
    ): Tensor2[A, B, DType] = 
      new Tensor2[A, B, DType](new Shape2[A, B](t.shape.dim1, t.shape.dim2), t.repr.add(y.repr), t.dtype)

    def rowsum: Tensor1[A, DType] =
      new Tensor1[A, DType](new Shape1[A](t.shape.dim1), t.repr.sum(dim = 0), t.dtype)

    def colsum: Tensor1[B, DType] =
      new Tensor1[B, DType](new Shape1[B](t.shape.dim2), t.repr.sum(dim = 1), t.dtype)

    def rowmean: Tensor1[A, Float32] =
      val newt : torch.Tensor[Float32] = torch.mean(t.repr.to(float32), dim = 0).to(float32)
      new Tensor1[A, Float32](new Shape1[A](t.shape.dim1), newt, float32)

    def colmean: Tensor1[B, Float32] =
      val newt : torch.Tensor[Float32] = torch.mean(t.repr.to(float32), dim = 1).to(float32)
      new Tensor1[B, Float32](new Shape1[B](t.shape.dim2), newt, float32)

    def colArgmax : Tensor1[B, Int32] =
      new Tensor1[B, Int32](new Shape1[B](t.shape.dim2), t.repr.argmax(dim = 0).to(int32), int32)
    
    def rowArgmax : Tensor1[A, Int32] =
      new Tensor1[A, Int32](new Shape1[A](t.shape.dim1), t.repr.argmax(dim = 1).to(int32), int32)

    def transpose(
      x: Tensor2[A, B, DType]
  ): Tensor2[B, A, DType] =
    new Tensor2[B, A, DType](new Shape2[B, A](x.shape.dim2, x.shape.dim1), x.repr.transpose(0, 1), x.dtype)
