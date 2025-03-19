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

import shapeful.nn.Transformation
import shapeful.nn.Relu
import shapeful.nn.Softmax
import shapeful.nn.|>

import torchvision.datasets.MNIST
import shapeful.distributions.Normal
import torch.Float32
import torch.DType.float32
import torch.indexing.Slice
import shapeful.tensor.Shape
import shapeful.nn.AffineTransformation


object MNistExample:

    sealed trait Data
    trait TrainData extends Data
    trait TestData extends Data

    type ImageFeatures = "image"
    val imageShape = Shape[ImageFeatures](28 * 28)
    type HiddenFeatures = "hidden"
    val hiddenShape = Shape[HiddenFeatures](100)
    type Output = "output"
    val Output = Shape[Output](10)


    def model (
        w1: Tensor2[ImageFeatures, HiddenFeatures], 
        b1: Tensor1[HiddenFeatures], 
        w2: Tensor2[HiddenFeatures, Output], 
        b2 : Tensor1[Output]
        )(x: Tensor2[Data, ImageFeatures]) 
        : Tensor2[Data, Output] = {
            val dataTrans = 
                AffineTransformation(w1, b1) 
                |> Relu() 
                |> AffineTransformation(w2, b2) 
                |> Softmax()
            dataTrans(x)
    }   


    def loss(y: Tensor2[Data, Output], yHat: Tensor2[Data, Output]): Tensor1[Data] = {
        val yhatlog = (yHat.add(Tensor(Shape.empty, 0.00001f))).log
        val l = y.mul(yhatlog)
        val m1 : Tensor0 = Tensor(Shape.empty, -1f)
        l.sum[Output].mul(m1)
    }

    def toOneHot(labels: Tensor1[Data]): Tensor2[Data, Output] = {
        val oneHotShape = Shape[Data, Output](labels.dim[Data], Output.dim[Output])
        val oneHot = Tensor(oneHotShape, 0)
        for i <- 0 until labels.dim[Data] do {
            val label = labels[Data](i).item.toInt//.select(Shape[Data](i)).item.toInt
            oneHot.update(Tuple2(i, label), 1) 
        }
        oneHot
    }
    
    def accuracy(y: Tensor2[Data, Output], yHat: Tensor2[Data, Output]): Float = {
        val predicted = yHat.argmax[Output]
        val label = y.argmax[Output]
        var correct : Int = 0
        for i <- 0 until y.dim[Data] do {
            
            if predicted[Data](i).item == label[Data](i).item  then
                correct += 1
        }
        correct / y.dim[Data].toFloat
    }

    def train() : Unit = 

        val mnist = MNIST(new java.io.File("./data").toPath(), train = true, download = true)
        val imageTensors = mnist.features.reshape(60000, 28 * 28)
        val labelsTensor : torch.Tensor[Float32] = mnist.targets.to(float32)

        val dataShape = Shape[Data](60000)
 
        val allimages : Tensor2[Data, ImageFeatures] = new Tensor(dataShape ++ imageShape, imageTensors)
        val alllabels : Tensor1[Data] = new Tensor(dataShape, labelsTensor)

        val (trainImages, valImages) = allimages.split(50000)
        val (trainLabels, valLabels) = alllabels.split(50000)
        val yTrain = toOneHot(trainLabels)
        val yVal = toOneHot(valLabels)

        val imageHiddenShape = imageShape ++ hiddenShape
        val hiddenOutputShape = hiddenShape ++ Output

        val w1 = Normal(imageHiddenShape, Tensor(imageHiddenShape, 0f), Tensor(imageHiddenShape, 0.01f)).sample().copy(requiresGrad = true)
        val b1 = Normal(hiddenShape, Tensor(hiddenShape, 0f), Tensor(hiddenShape, 0.01f)).sample().copy(requiresGrad = true)
        val w2 = Normal(hiddenOutputShape, Tensor(hiddenOutputShape, 0f), Tensor(hiddenOutputShape, 0.01f)).sample().copy(requiresGrad = true)
        val b2 = Normal(Output, Tensor(Output, 0f), Tensor(Output, 0.01f)).sample().copy(requiresGrad = true)
     
        val outputs = model(w1, b1, w2, b2)(trainImages)
      

        def f(images : Tensor2[Data, ImageFeatures], y : Tensor2[Data, Output])(
            w1: Tensor2[ImageFeatures, HiddenFeatures], 
            b1: Tensor1[HiddenFeatures], 
            w2: Tensor2[HiddenFeatures, Output], 
            b2 : Tensor1[Output]
            ) : Tensor0 = {
                val yHat = model(w1, b1, w2, b2)(images)
                loss(y, yHat).sum[Data]
            }

        val df = Autodiff.deriv(f(trainImages, yTrain))

        
        GradientOptimizer(-0.00001).optimize(df, (w1, b1, w2, b2)).zipWithIndex.take(1000).foreach{
            case ((w1, b1, w2, b2), i) => {
                if i % 10 == 0 then 
                    sys.runtime.gc()
                    println("Loss: " + f(trainImages, yTrain)(w1, b1, w2, b2).item)
                    val pred = model(w1, b1, w2, b2)(trainImages)
                    println("Training accuracy: " + accuracy(yTrain, pred))
                    val predTest = model(w1, b1, w2, b2)(valImages)
                    println("Validation accuracy: " + accuracy(yVal, predTest))
            }
        }


@main 
def runMnistExample() : Unit = 
    MNistExample.train()