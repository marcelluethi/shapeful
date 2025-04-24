package shapeful.examples

import shapeful.autodiff.Params
import torchvision.datasets.MNIST
import torch.Float32
import torch.DType.float32
import shapeful.optimization.GradientOptimizer
import shapeful.tensor.Tensor2
import shapeful.tensor.Variable2
import shapeful.tensor.Variable0
import shapeful.tensor.Tensor2Ops.*
import shapeful.tensor.TensorOps.*
import shapeful.tensor.Tensor1Ops.*
import shapeful.tensor.Tensor0
import shapeful.tensor.Tensor1
import torch.Int32
import shapeful.tensor.Shape
import shapeful.tensor.->>
import shapeful.distributions.Normal
import shapeful.autodiff.Autodiff
import shapeful.tensor.Tensor
import torch.DType.int32
import shapeful.tensor.Variable1
import shapeful.nn.*
import shapeful.tensor.Tensor3
import shapeful.tensor.{Tensor4, Variable4}
import org.bytedeco.pytorch.Slice

object MNistCNNExample:

    def model(x : Tensor4["data", "Value",  "imgHeight", "imgWidth", Float32])(params : Params) : Tensor2["data", "output", Float32] =

        val m = 
            Conv2D["Value", "imgHeight", "imgWidth", "Channels",  "imgHeight", "imgWidth"](
                params.get[Variable4["Channels", "Value", "imgHeight", "imgWidth"]]("convWeight"),
                params.get[Variable1["Channels"]]("convBias"),
                1, 
                0
            ).andThen(
                MaxPooling["Channels", "imgHeight", "imgWidth", "Channels", "imgHeight", "imgWidth"](2, 2, 0)
            )
            .andThen(
                Flatten4["Channels", "imgHeight", "imgWidth", "feature", Float32]()
            ).andThen(            
            AffineTransformation(
                    params.get[Variable2["feature", "hidden1"]]("w1"),
                    params.get[Variable1["hidden1"]]("b1")) 
            ).andThen( 
            Relu()
            ).andThen(
            AffineTransformation(
                params.get[Variable2["hidden1", "output"]]("w2"),
                params.get[Variable1["output"]]("b2")) 
            ).andThen(
                Softmax()
            )
        m(x)


    def loss(y : Tensor2["data", "output", Float32], yHat : Tensor2["data", "output", Float32]) : Tensor1["data", Float32] =
        val yhatlog = yHat.add(Tensor0(0.00001f)).log
        y.mul(yhatlog).mul( Tensor0(-1f)).sum["output"]


    def f(x : Tensor4["data",  "Value", "imgHeight", "imgWidth", Float32], y : Tensor2["data", "output", Float32])(params : Params) : Tensor0[Float32] =
        val yHat = model(x)(params)
        loss(y, yHat).mean

    def toOneHot(labels: Tensor1["data", Int32]): Tensor2["data", "output", Float32] = {
        val oneHotShape = Shape("data" ->> labels.shape.dim1, "output" ->> 10)
        val oneHot = Tensor2(oneHotShape, 0)
        for i <- 0 until labels.shape.dim1 do {
            val label = labels(i).item
            oneHot.update(i, label, Tensor0(1))
        }
        oneHot
    }

    def accuracy(y: Tensor2["data", "output", Float32], yHat: Tensor2["data", "output", Float32]): Float = {
        val predicted = yHat.argmax["output"]
        val label = y.argmax["output"]
        var correct : Int = 0
        for i <- 0 until y.shape.dim1 do {
            if predicted(i).item == label(i).item  then
                correct += 1
        }
        correct / y.shape.dim1.toFloat
    }

    def train(data : Tensor4["data",  "Value", "imgHeight", "imgWidth", Float32], target : Tensor2["data", "output", Float32]) : Unit =
        val nChannels = 10
        val nHidden = 50
        val convWeightShape = Shape("Channels" ->> nChannels, "Value" ->> 1, "imgHeight" ->> 3, "imgWidth" ->> 3)
        val convBiasShape = Shape("Channels" ->> nChannels)
        val w1Shape = Shape("features" ->> (13 * 13 * nChannels), "hidden1" ->> nHidden)
        val b1Shape = Shape("hidden1" ->> nHidden)
        val w2Shape = Shape("hidden1" ->> nHidden, "output" ->> 10)        
        val outputShape = Shape("output" ->> 10)

        val params = Params(Map(
            "convWeight" -> Normal(Tensor4(convWeightShape, 0.0), 
                            Tensor4(convWeightShape, math.sqrt(2.0 / (1 * 3 * 3)).toFloat)).sample().toVariable,
            "convBias" -> Tensor1(convBiasShape, 0f).toVariable,
            "w1" -> Normal(Tensor2(w1Shape, 0.0), 
                    Tensor2(w1Shape, math.sqrt(2.0 / (26 * 26 * 10)).toFloat)).sample().toVariable,
            "b1" -> Tensor1(b1Shape, 0f).toVariable,
            "w2" -> Normal(Tensor2(w2Shape, 0.0), 
                    Tensor2(w2Shape, math.sqrt(2.0 / 50).toFloat)).sample().toVariable,
            "b2" -> Tensor1(outputShape, 0f).toVariable,
        ))


        val df = Autodiff.deriv(f(data, target))

        val xs = GradientOptimizer(-0.1).optimize(df, params).zipWithIndex.take(1000)
        .foreach{
            case (params, i) => {
                if i % 10 == 0 then
                    sys.runtime.gc()
                    val yhat = model(data)(params)
                    println("Loss: " + loss(target, yhat).mean.item)
                    println("Accuracy: " + accuracy(target, yhat))
            }
        }  

@main
def runMnistCnnExample() : Unit =
    import torch.indexing.---
    import torch.indexing.Slice
    val mnist = MNIST(new java.io.File("./data").toPath(), train = true, download = true)
    val nImages = 10000
    val imageTensors = mnist.features.reshape(60000, 1, 28, 28)(Slice(0, nImages), ---, ---, ---)
    
    val labelsTensor : torch.Tensor[Int32] = mnist.targets.to(int32)

    val dataShape = Shape("data" ->> nImages, "Value" ->> 1, "imgHeight" ->> 28, "imgWidth" ->> 28)
    val labelShape = Shape("data" ->> nImages)
    val allimages = new Tensor4(dataShape, imageTensors)
    val alllabels = new Tensor1(labelShape, labelsTensor, dtype = int32)

    val yVal = MNistExample.toOneHot(alllabels)
    MNistCNNExample.train(allimages, yVal)