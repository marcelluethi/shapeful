package examples.plotting

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.distributions.Normal
import shapeful.autodiff.* 
import shapeful.optimization.GradientDescent
import shapeful.plotting.SeabornPlots
import shapeful.distributions.MVNormal
import shapeful.inference.AffineFlow
import examples.advanced.CustomPyTree.Output
import shapeful.inference.FromTensor1

object BayesianLinearRegression extends App:

    // Configure matplotlib for X11 display
    SeabornPlots.setupX11Display()
    
    type Feature = "feature"
    type Sample = "sample"
    type Latent = "Latent"

    val trueWeight = Tensor0(3.0f)
    val trueBias = Tensor0(-1.0f)
    val trueSigma = Tensor0(0.1f)
    val numSamples = 100

    // 1. Create synthetic dataset
    val X = Tensor1[Sample](Range(0, 100, 1).map(_.toFloat / 100f).toSeq)
    val noise = Normal(Tensor.zeros(X.shape), Tensor.ones(X.shape) * trueSigma).sample()
    val y = X.vmap[VmapAxis=Sample](sample => 
        trueWeight * sample + trueBias
    ) + noise

    // model params
    case class ModelParams(
        weight: Tensor0,
        bias: Tensor0, 
        logSigma : Tensor0
    ) derives TensorTree, ToPyTree

    given [L <: Label] : FromTensor1[L, ModelParams] with
        def convert(tensor: Tensor1[L]): ModelParams = 
            ModelParams(
                weight = tensor.at(Tuple1(0)).get,
                bias = tensor.at(Tuple1(1)).get,
                logSigma = tensor.at(Tuple1(2)).get 
            )

    // define the model 
    val priorW = Normal[EmptyTuple](Tensor0(2.0f), Tensor0(10.0f))
    val priorB = Normal[EmptyTuple](Tensor0(1.0f), Tensor0(10.0f))
    val priorSigma = Normal[EmptyTuple](Tensor0(0.1f), Tensor0(10.0f))
    
    def prior(params : ModelParams) = 
        priorB.logpdf(params.bias) + priorW.logpdf(params.weight) + priorSigma.logpdf(params.logSigma)

    def likelihood(x : Tensor1[Sample], y : Tensor1[Sample])(params : ModelParams): Tensor0 =        
        val sigma = params.logSigma.exp  // Convert log-sigma to sigma
        Normal(
            x * params.weight + params.bias, 
            Tensor.ones(x.shape) * sigma
        ).logpdf(y).sum 

    def posterior(x : Tensor1[Sample], y : Tensor1[Sample])(params : ModelParams) : Tensor0 = 
        prior(params) + likelihood(x, y)(params)
        

    val baseDistribution = MVNormal.standardNormal[Latent](Shape1(3)) // 3D latent space for weight, bias, logSigma
    val flow = AffineFlow[Latent, Output]()

    val nf = inference.NormalizingFlow(baseDistribution, flow)
    val elbo = nf.elbo(1000, posterior(X, y))
   
    // inference with corrected learning rate
    val optimizer = GradientDescent(-1.5e-4)  
    val grad = Autodiff.grad(elbo)

    val initialParams = AffineFlow.AffineFlowParams[Latent, Output](
      weight = Tensor.eye(Shape2(3, 3)),
      bias = Tensor.zeros(Shape1(3))
    )

    // Collect optimization trajectory
    val trajectory = optimizer.optimize(grad, initialParams)
        .take(750)
        .zipWithIndex
        .map { case (params, iter) =>
            if (iter % 100 == 0) then
                println(s"Iteration $iter: loss =${elbo(params).toFloat}")
            params
        }
        .toSeq

    val lastParams = trajectory.last

    val sampledModelparams = nf.generate(1000)(lastParams)
    val meanWeight = sampledModelparams.foldLeft(Tensor0(0f))(
        (acc, e) => acc + e.weight
      ) / Tensor0(sampledModelparams.size)

    println("estimated weight " + meanWeight)

    val meanBias = sampledModelparams.foldLeft(Tensor0(0f))(
        (acc, e) => acc + e.bias
      ) / Tensor0(sampledModelparams.size)

    println("estimated bias " + meanBias)