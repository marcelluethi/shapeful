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
import shapeful.tensor.Tensor0
import shapeful.tensor.Tensor1
import shapeful.tensor.Tensor1Ops.mean
import torch.Int32
import shapeful.tensor.Shape
import shapeful.tensor.~>
import shapeful.distributions.Normal
import shapeful.autodiff.deriv
import shapeful.tensor.Tensor
import torch.DType.int32
import shapeful.tensor.Variable1
import shapeful.nn.Activation

object MNistExample:


    /**
     * definition of neural network
     * */
    def layer1(params : Params) : Tensor2["data", "features", Float32] => Tensor2["data", "hidden1", Float32] =
        x =>
            val w1  = params.get[Variable2["features", "hidden1"]]("w1")
            val b1 = params.get[Variable1["hidden1"]]("b1")
            Activation.relu(x.matmul(w1).addToCols(b1))

    def layer2(params : Params) : Tensor2["data", "hidden1", Float32] => Tensor2["data", "output", Float32] =
        x =>
            val w2  = params.get[Variable2["hidden1", "output"]]("w2")
            val b2 =  params.get[Variable1["output"]]("b2")
            Activation.softmax(x.matmul(w2).addToCols(b2))

    def model(x : Tensor2["data", "features", Float32])(params : Params) : Tensor2["data", "output", Float32] =
        val l1 = layer1(params)
        val l2 = layer2(params)
        l1.andThen(l2)(x)

    def loss(y : Tensor2["data", "output", Float32], yHat : Tensor2["data", "output", Float32]) : Tensor1["data", Float32] =
        val yhatlog = yHat.add(Tensor0(0.00001f)).log
        val l = y.mul(yhatlog)
        val m1 : Tensor0[Float32]= Tensor0(-1f)
        val theloss = l.sum["output"].mul(m1)
        theloss

    def f(x : Tensor2["data", "features", Float32], y : Tensor2["data", "output", Float32])(params : Params) : Tensor0[Float32] =
        val yHat = model(x)(params)
        loss(y, yHat).mean

    def toOneHot(labels: Tensor1["data", Int32]): Tensor2["data", "output", Float32] = {
        val oneHotShape = Shape("data" ~> labels.shape.dim1, "output" ~> 10)
        val oneHot = Tensor2(oneHotShape, 0)
        for i <- 0 until labels.shape.dim1 do {
            val label = labels(i).item
            oneHot.update(i, label, 1)
        }
        oneHot
    }

    def accuracy(y: Tensor2["data", "output", Float32], yHat: Tensor2["data", "output", Float32]): Float = {
        val predicted = yHat.rowArgmax
        val label = y.rowArgmax
        var correct : Int = 0
        for i <- 0 until y.shape.dim1 do {
            if predicted(i).item == label(i).item  then
                correct += 1
        }
        correct / y.shape.dim1.toFloat
    }

    def train(data : Tensor2["data", "features", Float32], target : Tensor2["data", "output", Float32]) : Unit =

        val w1Shape = Shape("features" ~> (28 * 28), "hidden1" ~> 50)
        val b1Shape = Shape("hidden1" ~> 50)
        val w2Shape = Shape("hidden1" ~> 50, "output" ~> 10)        
        val outputShape = Shape("output" ~> 10)
        

        val params = Params(Map(
            "w1" -> Normal(Tensor2(w1Shape, 0f), Tensor2(w1Shape, 0.01f)).sample().toVariable,
            "b1" -> Normal(Tensor1(b1Shape, 0f), Tensor1(b1Shape, 0.01f)).sample().toVariable,
            "w2" -> Normal(Tensor2(w2Shape, 0f), Tensor2(w2Shape, 0.01f)).sample().toVariable,
            "b2" -> Normal(Tensor1(outputShape, 0f), Tensor1(outputShape, 0.01f)).sample().toVariable,
        ))


        val df = deriv(f(data, target))

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
def runMnistExample() : Unit =
    val mnist = MNIST(new java.io.File("./data").toPath(), train = true, download = true)
    val imageTensors = mnist.features.reshape(60000, 28 *28)
    val labelsTensor : torch.Tensor[Int32] = mnist.targets.to(int32)

    val dataShape = Shape("data" ~> 60000, "features" ~> (28 * 28))
    val labelShape = Shape("data" ~> 60000)
    val allimages = new Tensor2(dataShape, imageTensors)
    val alllabels = new Tensor1(labelShape, labelsTensor, dtype = int32)

    val yVal = MNistExample.toOneHot(alllabels)
    MNistExample.train(allimages, yVal)


