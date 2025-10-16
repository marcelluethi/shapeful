package examples.inference

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
import shapeful.inference.flows.AffineCouplingFlow.initParams
import shapeful.distributions.HalfNormal
import examples.advanced.CustomPyTree.Output
import examples.advanced.CustomPyTree.Hidden
import shapeful.random.Random

object BayesianLinearRegression extends App:

  // Create a random key for reproducible randomness
  val key = Random.Key(42)

  type Feature = "feature"
  type Sample = "sample"
  type Latent = "latent"

  val trueWeight = Tensor0(3.0f)
  val trueBias = Tensor0(-0.0f)
  val trueSigma = Tensor0(0.2f)
  val numSamples = 100

  // 1. Create synthetic dataset
  val (key1, key2) = key.split2()
  val X = Tensor1[Sample](Range(0, 100, 1).map(_.toFloat / 100f).toSeq)
  val noise = Normal(Tensor.zeros(X.shape), Tensor.ones(X.shape) * trueSigma).sample(key1)
  val y = X.vmap(Axis[Sample], sample => trueWeight * sample + trueBias) + noise

  // model params
  case class ModelParams(
      weight: Tensor0,
      bias: Tensor0,
      sigma: Tensor0
  ) derives TensorTree,
        ToPyTree

  given [L <: Label]: FromTensor1[L, ModelParams] with
    def convert(tensor: Tensor1[L]): ModelParams =

      ModelParams(
        weight = tensor.at(Tuple1(0)).get,
        bias = tensor.at(Tuple(1)).get, // Scale to match prior
        sigma = tensor.at(Tuple1(2)).get.exp.clamp(0.01f, 10.0f)
      )

  // define the model
  val priorW = Normal(trueWeight, Tensor0(5.0f))
  val priorB = Normal(trueBias, Tensor0(5.0f))
  val priorSigma = HalfNormal(Tensor0(1f))

  def prior(params: ModelParams) =
    priorB.logpdf(params.bias) + priorW.logpdf(params.weight) + priorSigma.logpdf(params.sigma)

  def likelihood(x: Tensor1[Sample], y: Tensor1[Sample])(params: ModelParams): Tensor0 =

    Normal(
      x * params.weight + params.bias,
      Tensor.ones(x.shape) * params.sigma
    ).logpdf(y).sum

  def posterior(x: Tensor1[Sample], y: Tensor1[Sample])(params: ModelParams): Tensor0 =
    prior(params) + likelihood(x, y)(params)

  val baseDistribution = MVNormal.standard[Latent](
    Shape1(3)
  ) // Tensor.zeros(Shape1(3)), Tensor.eye(Shape2(3, 3)) * Tensor0(0.1f)) // 3D latent space for weight, bias, logSigma

  val mask1 = Tensor1[Latent](Seq(1.0f, 0.0f, 1.0f)) // Example mask for 3D latent space
  val flow1 = AffineCouplingFlow[Latent](mask1)
  val mask2 = Tensor1[Latent](Seq(0.0f, 1.0f, 1.0f)) // Another mask for the second flow
  val flow2 = AffineCouplingFlow[Latent](mask2)

  case class CompositeParams(
      flow1Params: AffineCouplingFlow.Params[Latent],
      flow2Params: AffineCouplingFlow.Params[Latent]
  ) derives TensorTree,
        ToPyTree
  object CompositeParams:
    def initialParams(): CompositeParams =
      val (initKey1, initKey2) = key2.split2()
      CompositeParams(
        flow1Params = AffineCouplingFlow.initParams[Latent](mask = mask1, hidden_dim = 3, key = initKey1),
        flow2Params = AffineCouplingFlow.initParams[Latent](mask = mask2, hidden_dim = 3, key = initKey2)
      )

  // make a composite flow
  class CompositeAffineCouplingFlow extends Flow[Latent, Latent, CompositeParams]:
    def forwardSample(tensor: Tensor1[Latent])(params: CompositeParams): Tensor1[Latent] =
      val intermediate = flow1.forwardSample(tensor)(params.flow1Params)
      flow2.forwardSample(intermediate)(params.flow2Params)

  val flow = CompositeAffineCouplingFlow()
  val initialParams = CompositeParams.initialParams()

  // // val flow = IdentityFlow[Latent]() // Use IdentityFlow for simplicity
  // val initialParams = IdentityFlow.initialParams

  val nf = NormalizingFlow(baseDistribution, flow)
  val (elboKey, optimizerKey) = key2.split2()
  val elbo = nf.elbo(100, posterior(X, y), key = elboKey)
  val grad = Autodiff.grad(elbo)

  val optimizer = GradientDescent(-1e-4f)
  val trajectory = optimizer.optimize(grad, initialParams).zipWithIndex.map { (params, iter) =>
    if iter % 50 == 0 then
      // println("Iteration: " + iter)
      // val loss = elbo(params)
      // //val realParamSample = nf.generate(1)(params).head
      // //println("real params: " + realParamSample)
      // println("in iteration: " + iter +"\n ==============")
      // println("params: " + params)
      // println("gradients: " + grad(params))

      // println(s"Loss: ${loss.toFloat}")

      // // val asParams = nf.generate(1)(params).head
      // // println("model params: " + asParams)

      // Debug: compute individual components before clamping
      val (baseKey, _) = optimizerKey.split2()
      val baseSamples = baseDistribution.sampleBatch["Sample"](10, key = baseKey)
      val transformedSamples = flow.forward(baseSamples)(params)
      val logdet = baseSamples.vmap(
        Axis["Sample"],
        { sample =>
          flow.logDetJacobian(sample)(params)
        }
      )
      val realParams = flow.forward(baseSamples)(params)
      for realParam <- realParams.unstack(Axis["Sample"]) do
        // println("real param: " + realParam)
        val modelParams = summon[FromTensor1[Latent, ModelParams]].convert(realParam.asInstanceOf[Tensor1[Latent]])
        println("model params: " + modelParams)

      val baseLogProb = baseSamples.vmap(Axis["Sample"], baseDistribution.logpdf)
      val targetLogProb = transformedSamples.vmap(
        Axis["Sample"],
        t =>
          val modelParam = summon[FromTensor1["latent", ModelParams]].convert(t.asInstanceOf[Tensor1["latent"]])
          posterior(X, y)(modelParam)
      )

      println(s"Base log prob range: ${baseLogProb.min.toFloat} to ${baseLogProb.max.toFloat}")
      println(s"Target log prob range: ${targetLogProb.min.toFloat} to ${targetLogProb.max.toFloat}")
      println(s"Log det range: ${logdet.min.toFloat} to ${logdet.max.toFloat}")

      val rawLogProbs: Tensor1["Sample"] = targetLogProb - baseLogProb + logdet
      println(s"Raw log probs range: ${rawLogProbs.min.toFloat} to ${rawLogProbs.max.toFloat}")

      val loss = elbo(params)
      println(s"Loss: ${loss.toFloat}")
    params
  }

  val lastParams = trajectory.take(500).toSeq.last

  val (generateKey, _) = optimizerKey.split2()
  val sampledModelparams = nf.generate(1000, key = generateKey)(lastParams)
  val meanWeight =
    sampledModelparams.foldLeft(Tensor0(0f))((acc, e) => acc + e.weight) / Tensor0(sampledModelparams.size)

  println("estimated weight " + meanWeight)

  val meanBias = sampledModelparams.foldLeft(Tensor0(0f))((acc, e) => acc + e.bias) / Tensor0(sampledModelparams.size)

  println("estimated bias " + meanBias)
