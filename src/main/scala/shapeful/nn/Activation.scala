package shapeful.nn

import shapeful.tensor.Tensor2
import torch.Float32
import torch.DType.float32


class Relu[A <: Singleton]() extends Transformation2[A, A, Float32]:
    override def apply[Data <: Singleton](x : Tensor2[Data, A, Float32]) : Tensor2[Data, A, Float32] =
        val relu = torch.nn.ReLU(false)
        val newtensor = relu(x.repr.to(float32))
        new Tensor2(x.shape, newtensor, float32)

class Softmax[A <: Singleton]() extends Transformation2[A, A, Float32]:
    override def apply[Data <: Singleton](x : Tensor2[Data, A, Float32]) : Tensor2[Data, A, Float32] =
        val featureInd = 1
        val newtensor = torch.softmax(x.repr.to(float32), dim = featureInd, dtype = torch.float32)
        new Tensor2(x.shape, newtensor, float32)
