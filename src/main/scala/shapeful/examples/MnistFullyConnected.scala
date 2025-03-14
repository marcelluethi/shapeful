package shapeful.examples

import shapeful.tensor.Dimension.Symbolic
import shapeful.tensor.Tensor.{Tensor0, Tensor1, Tensor2}

import shapeful.tensor.TensorOps.* 
import shapeful.tensor.Tensor0Ops.* 
import shapeful.tensor.Tensor2Ops.* 

import shapeful.autodiff.Autodiff
import shapeful.tensor.TensorTupleOps
import shapeful.optimization.GradientOptimizer
import shapeful.tensor.Dimension
import shapeful.tensor.Tensor
import shapeful.nn.Activation.relu
import shapeful.nn.Activation.softmax
import torchvision.datasets.MNIST
import shapeful.distributions.Normal
import torch.Float32
import torch.DType.float32
import torch.indexing.Slice
import shapeful.tensor.Shape


object MNistExample:

    sealed trait Data
    trait TrainData extends Data
    trait TestData extends Data

    type Image = "image"
    val imageShape = Shape[Image](28 * 28)

    type Hidden = "hidden"
    val hiddenShape = Shape[Hidden](100)

    type Output = "output"
    val outputShape = Shape[Output](10)


    def model[A <: Data] (
        w1: Tensor2[Image, Hidden], 
        b1: Tensor1[Hidden], 
        w2: Tensor2[Hidden, Output], 
        b2 : Tensor1[Output]
        )(x: Tensor2[A, Image]) 
        : Tensor2[A, Output] = {
            val h1 = relu(x.matmul(w1)).addAlongDim(b1)
            val out = h1.matmul(w2).addAlongDim(b2)
            softmax(out)
    }


    def loss[D <: Data](y: Tensor2[D, Output], yHat: Tensor2[D, Output]): Tensor1[D] = {
        val yhatlog = (yHat.add(Tensor(Shape.empty, 0.00001f))).log
        val l = y.mul(yhatlog)
        val m1 : Tensor0 = Tensor(Shape.empty, -1f, requiresGrad = false)
        l.sum[Output].mul(m1)
    }

    def toOneHot[A <: Data](labels: Tensor1[A]): Tensor2[A, Output] = {
        val oneHotShape = Shape[A, Output](labels.dim[A], outputShape.dim[Output])
        val oneHot = Tensor(oneHotShape, 0)
        for i <- 0 until labels.dim[A] do {
            val label = labels.get(Tuple1(i)).toInt
            oneHot.update(Tuple2(i, label), 1) 
        }
        oneHot
    }
    
    def accuracy[D <: Data](y: Tensor2[D, Output], yHat: Tensor2[D, Output]): Float = {
        val predicted = yHat.argmax[Output]
        val label = y.argmax[Output]
        var correct : Int = 0
        for i <- 0 until y.dim[D] do {
            
            if predicted.get(Tuple1(i)) == label.get(Tuple1(i))  then
                correct += 1
        }
        correct / y.dim[D].toFloat
    }

    def train() : Unit = 

        val mnist = MNIST(new java.io.File("./data").toPath(), train = true, download = true)
        val imageTensors = mnist.features.reshape(60000, 28 * 28)
        val labelsTensor : torch.Tensor[Float32]=       mnist.targets.to(float32)

        val dataShape = Shape[Data](60000)
        val dataImageShape = Shape[Data, Image](dataShape.dim[Data], imageShape.dim[Image])

        val allimages : Tensor2[Data, Image] = new Tensor(dataImageShape, imageTensors)
        val alllabels : Tensor1[Data] = new Tensor(dataShape, labelsTensor)

        val (trainImages, testImags) = allimages.split(30000)
        val (trainLabels, testLabels) = alllabels.split(30000)
        val yTrain = toOneHot(trainLabels)
        val yTest = toOneHot(testLabels)

        val imageHiddenShape = Shape[Image, Hidden](imageShape.dim[Image], hiddenShape.dim[Hidden])
        val hiddenOutputShape = Shape[Hidden, Output](hiddenShape.dim[Hidden], outputShape.dim[Output])

        val w1 = Normal(imageHiddenShape, Tensor(imageHiddenShape, 0f), Tensor(imageHiddenShape, 0.01f)).sample().copy(requiresGrad = true)
        val b1 = Normal(hiddenShape, Tensor(hiddenShape, 0f), Tensor(hiddenShape, 0.01f)).sample().copy(requiresGrad = true)
        val w2 = Normal(hiddenOutputShape, Tensor(hiddenOutputShape, 0f), Tensor(hiddenOutputShape, 0.01f)).sample().copy(requiresGrad = true)
        val b2 = Normal(outputShape, Tensor(outputShape, 0f), Tensor(outputShape, 0.01f)).sample().copy(requiresGrad = true)
     
        val outputs = model(w1, b1, w2, b2)(trainImages)
      

        def f[D <: Data](images : Tensor2[D, Image], y : Tensor2[D, Output])(
            w1: Tensor2[Image, Hidden], 
            b1: Tensor1[Hidden], 
            w2: Tensor2[Hidden, Output], 
            b2 : Tensor1[Output]
            ) : Tensor0 = {
                val yHat = model[D](w1, b1, w2, b2)(images)
                loss[D](y, yHat).sum[D]
            }

        val df = Autodiff.deriv(f(trainImages, yTrain))

        // an optimization loop (manually)

        
        GradientOptimizer(-0.00005).optimize(df, (w1, b1, w2, b2)).zipWithIndex.take(1000).foreach{
            case ((w1, b1, w2, b2), i) => {
                if i % 10 == 0 then 
                    sys.runtime.gc()
                    println("Loss: " + f(trainImages, yTrain)(w1, b1, w2, b2).item)
                    val pred = model(w1, b1, w2, b2)(trainImages)
                    println("TrainAccuracy: " + accuracy(yTrain, pred))
                    val predTest = model(w1, b1, w2, b2)(testImags)
                    println("TrainAccuracy: " + accuracy(yTest, predTest))
            }
        }


@main 
def runMnistExample() : Unit = 
    MNistExample.train()