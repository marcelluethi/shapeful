// package shapeful.examples

// import shapeful.autodiff.Params
// import shapeful.tensor.TensorExpr2
// import shapeful.tensor.Variable2
// import shapeful.tensor.Variable0
// import shapeful.tensor.Plus2Scalar
// import shapeful.tensor.MatMul2
// import shapeful.tensor.Diff2
// import shapeful.tensor.Sum
// import shapeful.tensor.TensorExpr0
// import shapeful.tensor.Mul2
// import shapeful.tensor.TensorExpr1
// import shapeful.tensor.DataTensor2
// import torchvision.datasets.MNIST
// import torch.Float32
// import torch.DType.float32
// import shapeful.autodiff.Autodiff
// import shapeful.optimization.GradientOptimizer

// object MNistExample:

//     /**
//      * definition of neural network
//      * */
//     def layer1(params : Params) : TensorExpr2["data", "features"] => TensorExpr2["data", "hidden1"] =
//         x =>
//             val w1  = params.read[Variable2["features", "hidden1"]]("w1")
//             val b1 = params.read[Variable0]("b1")
//             Plus2Scalar(MatMul2(x, w1), b1)

//     def layer2(params : Params) : TensorExpr2["data", "hidden1"] => TensorExpr2["data", "output"] =
//         x =>
//             val w2  = params.read[Variable2["hidden1", "output"]]("w2")
//             val b2 =  params.read[Variable0]("b2")
//             val xw2 : TensorExpr2["data", "output"] = MatMul2(x, w2)
//             Plus2Scalar(xw2, b2)

//     def model(x : TensorExpr2["data", "features"])(params : Params) : TensorExpr2["data", "output"] =
//         val l1 = layer1(params.read[Params]("layer1"))
//         val l2 = layer2(params.read[Params]("layer2"))
//         l1.andThen(l2)(x)

//     def loss(y : TensorExpr2["data", "output"], yHat : TensorExpr2["data", "output"]) : TensorExpr0 =
//         val diff = Diff2(y, yHat)
//         val diff2 = Mul2(diff, diff)
//         val sum = Sum(diff2)
//         sum

//     def f(x : TensorExpr2["data", "features"], y : TensorExpr2["data", "output"])(params : Params) : TensorExpr0 =
//         val yHat = model(x)(params)
//         loss(y, yHat)

//     def toOneHot(labels: TensorExpr1["data"]): TensorExpr2["data", "output"] = {
//         // val oneHotShape = Shape[Data, Output](labels.dim[Data], Output.dim[Output])
//         // val oneHot = Tensor(oneHotShape, 0)
//         // for i <- 0 until labels.dim[Data] do {
//         //     val label = labels[Data](i).item.toInt//.select(Shape[Data](i)).item.toInt
//         //     oneHot.update(Tuple2(i, label), 1)
//         // }
//         // oneHot
//         ???
//     }

//     def accuracy(y: DataTensor2["data", "output"], yHat: DataTensor2["data", "output"]): Float = {
//         // val predicted = yHat.argmax[Output]
//         // val label = y.argmax[Output]
//         // var correct : Int = 0
//         // for i <- 0 until y.dim[Data] do {

//         //     if predicted[Data](i).item == label[Data](i).item  then
//         //         correct += 1
//         // }
//         // correct / y.dim[Data].toFloat
//         ???
//     }

//     def train(data : DataTensor2["data", "features"], target : DataTensor2["data", "output"]) : Unit =

//         val params = Params(Map[String, Any]())

//         val fv = f(data, target)(params)
//         val df = Autodiff.deriv(f(data, target))

//         val xs = GradientOptimizer(-0.00001).optimize(df, params).zipWithIndex.take(1000)
//         //.foreach{
//         //     case (params, i) => {
//         //         if i % 100 == 0 then
//         //             sys.runtime.gc()
//         //             println("Loss: " + f(data, target)(params))
//         //     }
//         // }

//         // val mnist = MNIST(new java.io.File("./data").toPath(), train = true, download = true)
//         // val imageTensors = mnist.features.reshape(60000, 28, 28)
//         // val labelsTensor : torch.Tensor[Float32] = mnist.targets.to(float32)

//         // val dataShape = Shape[Data](60000)

//         // val allimages : Tensor3[Data, ImageHeight, ImageWidth] = new Tensor(dataShape ++ imageShape, imageTensors)
//         // val alllabels : Tensor1[Data] = new Tensor(dataShape, labelsTensor)

//         // val (trainImages, valImages) = allimages.split(50000)
//         // val (trainLabels, valLabels) = alllabels.split(50000)
//         // val yTrain = toOneHot(trainLabels)
//         // val yVal = toOneHot(valLabels)

//         // val imageHiddenShape = imageFlattendShape ++ hiddenShape
//         // val hiddenOutputShape = hiddenShape ++ Output

//         // val w1 = Normal(imageHiddenShape, Tensor(imageHiddenShape, 0f), Tensor(imageHiddenShape, 0.01f)).sample().copy(requiresGrad = true)
//         // val b1 = Normal(hiddenShape, Tensor(hiddenShape, 0f), Tensor(hiddenShape, 0.01f)).sample().copy(requiresGrad = true)
//         // val w2 = Normal(hiddenOutputShape, Tensor(hiddenOutputShape, 0f), Tensor(hiddenOutputShape, 0.01f)).sample().copy(requiresGrad = true)
//         // val b2 = Normal(Output, Tensor(Output, 0f), Tensor(Output, 0.01f)).sample().copy(requiresGrad = true)
//         // val wconv = Normal(Shape[Conv1KernelHeight, Conv1KernelWidth](3, 3), Tensor(Shape[Conv1KernelHeight, Conv1KernelWidth](3, 3), 0f), Tensor(Shape[Conv1KernelHeight, Conv1KernelWidth](3, 3), 0.01f)).sample().copy(requiresGrad = true)
//         // val outputs = model(wconv, w1, b1, w2, b2)(trainImages)

//         // def f(images : Tensor3[Data, ImageHeight, ImageWidth], y : Tensor2[Data, Output])(
//         //     wconv: Tensor2[Conv1KernelHeight, Conv1KernelWidth],
//         //     w1: Tensor2[ImageFeatures, HiddenFeatures],
//         //     b1: Tensor1[HiddenFeatures],
//         //     w2: Tensor2[HiddenFeatures, Output],
//         //     b2 : Tensor1[Output]
//         //     ) : Tensor0 = {
//         //         val yHat = model(wconv, w1, b1, w2, b2)(images)
//         //         loss(y, yHat).sum[Data]
//         //     }

//         // val df = Autodiff.deriv(f(trainImages, yTrain))

//         // GradientOptimizer(-0.00001).optimize(df, (wconv, w1, b1, w2, b2)).zipWithIndex.take(1000).foreach{
//         //     case ((wconv, w1, b1, w2, b2), i) => {
//         //         if i % 100 == 0 then
//         //             sys.runtime.gc()
//         //             println("Loss: " + f(trainImages, yTrain)(wconv, w1, b1, w2, b2).item)
//         //             val pred = model(wconv, w1, b1, w2, b2)(trainImages)
//         //             println("Training accuracy: " + accuracy(yTrain, pred))
//         //             val predTest = model(wconv,w1, b1, w2, b2)(valImages)
//         //             println("Validation accuracy: " + accuracy(yVal, predTest))
//         //     }
//         // }

// @main
// def runMnistExample() : Unit =

//     val data : DataTensor2["data", "features"] = ???
//     val target : DataTensor2["data", "output"] = ???

//     MNistExample.train(data, target)
