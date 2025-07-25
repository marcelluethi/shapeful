package examples.inference

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.distributions.Normal
import shapeful.autodiff.* 
import shapeful.optimization.GradientDescent
import shapeful.plotting.SeabornPlots
import shapeful.distributions.MVNormal
import shapeful.inference.{NormalizingFlow, FromTensor1}
import shapeful.inference.flows.Flow
import shapeful.inference.flows.IdentityFlow
import shapeful.inference.flows.AffineCouplingFlow
import shapeful.jax.Jax.PyAny

object BayesianLinearRegression extends App:
    
    type Feature = "feature"
    type Sample = "sample"
    type Latent = "latent"

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
    val priorW = Normal(Tensor0(2.0f), Tensor0(10.0f))
    val priorB = Normal(Tensor0(1.0f), Tensor0(10.0f))
    val priorSigma = Normal(Tensor0(0.1f), Tensor0(10.0f))
    
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
    
    // Set up composite affine coupling flow with 3 layers
    val hidden_dim = 32
    val input_dim = 3
    
    // Alternating mask patterns for better mixing
    val mask1 = Tensor1[Latent](Seq(1.0f, 0.0f, 1.0f)) // Transform bias, condition on weight & logSigma
    val mask2 = Tensor1[Latent](Seq(0.0f, 1.0f, 0.0f)) // Transform weight & logSigma, condition on bias
    val mask3 = Tensor1[Latent](Seq(1.0f, 0.0f, 1.0f)) // Transform bias again with updated conditioning
    
    // Create 3 coupling flows
    val flow1 = new AffineCouplingFlow[Latent](hidden_dim)
    val flow2 = new AffineCouplingFlow[Latent](hidden_dim)
    val flow3 = new AffineCouplingFlow[Latent](hidden_dim)
    
    // Initialize parameters for each flow
    val flow1Params = AffineCouplingFlow.initParams(input_dim, hidden_dim, mask1)
    val flow2Params = AffineCouplingFlow.initParams(input_dim, hidden_dim, mask2)
    val flow3Params = AffineCouplingFlow.initParams(input_dim, hidden_dim, mask3)
    
    // Create composite parameter structure
    case class CompositeFlowParams(
        flow1: AffineCouplingFlow.Params[Latent],
        flow2: AffineCouplingFlow.Params[Latent],
        flow3: AffineCouplingFlow.Params[Latent]
    ) derives TensorTree, ToPyTree
    
    // Composite flow that applies flows in sequence
    class CompositeFlow extends Flow[Latent, Latent, CompositeFlowParams]:
        def forwardSample(x: Tensor1[Latent])(params: CompositeFlowParams): Tensor1[Latent] =
            val x1 = flow1.forwardSample(x)(params.flow1)
            val x2 = flow2.forwardSample(x1)(params.flow2)
            val x3 = flow3.forwardSample(x2)(params.flow3)
            x3
        
        override def logDetJacobian(x: Tensor1[Latent])(params: CompositeFlowParams): Tensor0 =
            val x1 = flow1.forwardSample(x)(params.flow1)
            val x2 = flow2.forwardSample(x1)(params.flow2)
            
            val logDet1 = flow1.logDetJacobian(x)(params.flow1)
            val logDet2 = flow2.logDetJacobian(x1)(params.flow2)
            val logDet3 = flow3.logDetJacobian(x2)(params.flow3)

            val totalLogDet = logDet1 + logDet2 + logDet3
            totalLogDet.clamp(-15.0f, 15.0f) 

    val flow = new CompositeFlow()
    val flowParams = CompositeFlowParams(flow1Params, flow2Params, flow3Params)

    val nf = NormalizingFlow(baseDistribution, flow)
    val elbo = nf.elbo(1000, posterior(X, y))
   
    // inference with smaller learning rate for composite flow stability
    val optimizer = GradientDescent(-2e-5f)  // Reduced from -1e-4f
    val grad = Autodiff.grad(elbo)

    val initialParams = flowParams

    // Collect optimization trajectory
    val trajectory = optimizer.optimize(grad, initialParams)
        .take(1500)  // Increased from 750 for composite flow
        .zipWithIndex
        .map { case (params, iter) =>
            if (iter % 100 == 0) then
                println(s"Iteration $iter: loss = ${elbo(params).toFloat}")
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