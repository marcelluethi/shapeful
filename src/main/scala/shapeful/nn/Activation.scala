package shapeful.nn

import shapeful.tensor.Tensor2
import torch.Float32

// import shapeful.tensor.Tensor.Tensor2
// import shapeful.tensor.Tensor
// import scala.compiletime.{constValue, erasedValue, summonFrom}

// class Relu[A]() extends Transformation2[A, A]:
//     override def apply[Data](x : Tensor2[Data, A]) : Tensor2[Data, A] =
//         val relu = torch.nn.ReLU(false)
//         val newtensor = relu(x.stensor)
//         new Tensor(x.shape, newtensor)

// class Softmax[A]() extends Transformation2[A, A]:
//     override def apply[Data](x : Tensor2[Data, A]) : Tensor2[Data, A] =
//         val featureInd = 1
//         val newtensor = torch.softmax(x.stensor, dim = featureInd, dtype = torch.float32)
//         new Tensor(x.shape, newtensor)

object Activation:
    def relu[A <: Singleton](x : Tensor2["data", A, Float32]) : Tensor2["data", A, Float32] =
        val rlu = torch.nn.ReLU(false)
        val newtensor = rlu(x.repr)
        new Tensor2(x.shape, newtensor)


    def softmax[A <: Singleton](x : Tensor2["data", A, Float32]) : Tensor2["data", A, Float32] =
        val smx = torch.nn.Softmax(dim = 1)
        val newtensor = smx(x.repr)
        new Tensor2(x.shape, newtensor)
