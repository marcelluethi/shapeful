package shapeful.examples

import shapeful.tensor.Dimension.Symbolic
import shapeful.tensor.Tensor.{Tensor0, Tensor1, Tensor2}

import shapeful.tensor.TensorOps.* 
import shapeful.tensor.Tensor0Ops.* 
import shapeful.tensor.Tensor2Ops.* 

import shapeful.autodiff.Autodiff
import shapeful.tensor.TensorTupleOps
import shapeful.optimization.GradientDescent
import shapeful.tensor.Dimension
import shapeful.tensor.Tensor
import shapeful.nn.Activation.relu
import shapeful.nn.Activation.softmax
import torchvision.datasets.MNIST
import shapeful.distributions.Normal

object MNistExample:

    type Data = "data"
    given Dimension[Data] = Symbolic[Data](60000) // TODO replace with number of images

    type Image = "image"
    given Dimension[Image] = Symbolic[Image](28 * 28)

    type Hidden = "hidden"
    given Dimension[Hidden] = Symbolic[Hidden](100)

    type Output = "output"
    given Dimension[Output] = Symbolic[Output](10)

    type OutputLabel = "outputLabel"
    given Dimension[OutputLabel] = Symbolic[OutputLabel](1)

    def forward (
        w1: Tensor2[Image, Hidden], 
        b1: Tensor1[Hidden], 
        w2: Tensor2[Hidden, Output], 
        b2 : Tensor1[Output]
        )(x: Tensor2[Data, Image]) 
        : Tensor2[Data, Output] = {
            val h1 = relu(x.matmul(w1)).addAlongDim(b1)
            val out = h1.matmul(w2).addAlongDim(b2)
            softmax(out)
    }

    def loss(y: Tensor2[Data, Output], yHat: Tensor2[Data, Output]): Tensor1[Data] = {
        val ylog = y.log
        val l = yHat.mul(ylog)
        val m1 : Tensor0 = Tensor(-1f, requiresGrad = false)
        l.sum[Output].mul(m1)
    }

    def toOneHot(labels: Tensor2[Data, OutputLabel]): Tensor2[Data, Output] = {
        // val oneHot = Tensor[(Data, Output)](0)
        // for i <- 0 until labels.shape(0) do {
        //     val label = labels(i, 0).toInt
        //     oneHot(i, label) = 1f
        // }
        // oneHot
        ???
    }

    def train() : Unit = 
        // a sequence of images, represented as tensors
        
        val mnist = MNIST(new java.io.File("./data").toPath(), train = true, download = true)
        val imageTensors = mnist.features.reshape(60000, 28 * 28)
        val images : Tensor2[Data, Image] = new Tensor(imageTensors, imageTensors.shape.toList)
        
        println(mnist.targets)
        val labels : Tensor2[Data, OutputLabel] = new Tensor(mnist.targets.reshape(60000, 1).float, List(60000, 1)) 

        val w1 : Tensor2[Image, Hidden] = Normal[(Image, Hidden)](Tensor(0f), Tensor(0.01f)).sample()
        val b1 : Tensor1[Hidden] = Normal[Tuple1[Hidden]](Tensor(0f), Tensor(0.01f)).sample()
        val w2 : Tensor2[Hidden, Output] = Normal[(Hidden, Output)](Tensor(0f), Tensor(0.01f)).sample()
        val b2 : Tensor1[Output] = Normal[Tuple1[Output]](Tensor(0f), Tensor(0.01f)).sample()
        val outputs = forward(w1, b1, w2, b2)(images)

        //val l = loss(labels, outputs)



@main 
def runMnistExample() : Unit = 
    MNistExample.train()