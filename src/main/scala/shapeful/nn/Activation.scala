// package shapeful.nn

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
