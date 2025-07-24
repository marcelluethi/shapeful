package examples.inference

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.distributions.MVNormal
import shapeful.autodiff.{Autodiff, TensorTree, ToPyTree}
import shapeful.optimization.GradientDescent
import shapeful.inference.{AffineFlow, NormalizingFlow}
import shapeful.inference.AffineFlow.AffineFlowParams
import shapeful.inference.NormalizingFlow.Flow

/** Bayesian Inference with Normalizing Flows example.
  *
  * This example demonstrates:
  *   - Bayesian posterior approximation using normalizing flows
  *   - Learning a flow that transforms from prior to approximate posterior
  *   - Case class parameters with automatic TensorTree derivation
  *   - Variational inference with change of variables
  */
object NormalizingFlowExample extends App:

  println("=== Bayesian Inference with Normalizing Flows ===\n")

  // Define dimension labels
  type LatentParam = "latent_param" // Latent space for flow
  type LatentParam2 = "latent_param2" // Intermediate latent space

  type ModelParam = "model_param" // Model parameter space
  type Sample = "sample"

  val paramDim = 2 // Dimensionality of model parameters
  val numSamples = 1000

  // 2. True parameters (what we want to infer in Bayesian setting)
  println("Setting up True Model Parameters")
  val trueWeight = Tensor2[LatentParam, ModelParam](
    Seq(
      Seq(2.0f, 0.5f), // True parameter values we want to recover
      Seq(-0.3f, 1.5f) // via Bayesian inference
    )
  )
  val trueBias = Tensor1[ModelParam](Seq(1.0f, -0.5f))

  println(s"True model parameters:")
  println(s"  Weight:\n${trueWeight}")
  println(s"  Bias: ${trueBias}")
  println("(In real Bayesian inference, these would be unknown)")
  println()

  // 3. Define prior distribution p(θ) - our belief before seeing data
  println("3. Setting up Prior Distribution p(θ)")
  val priorMu = Tensor1[LatentParam](Seq(0.0f, 0.0f)) // Zero mean prior
  val priorCov = Tensor2[LatentParam, LatentParam](
    Seq(
      Seq(1.0f, 0.0f), // Unit variance, independent prior
      Seq(0.0f, 1.0f)
    )
  )
  val priorDist = MVNormal(priorMu, priorCov)
  println(s"Prior: p(z) = MVNormal(μ=${priorMu}, Σ=I)")
  println("This represents our initial belief about the latent variables")
  println()

  // 4. Define target posterior distribution p(θ|data) - what we want to approximate
  println("4. Setting up Target Posterior Distribution p(θ|data)")
  val posteriorMu = Tensor1[ModelParam](Seq(1.0f, -0.5f)) // Posterior mean (from "observed data")
  val posteriorCov = Tensor2[ModelParam, ModelParam](
    Seq(
      Seq(4.25f, 0.45f), // Posterior covariance (from "observed data")
      Seq(0.45f, 2.34f) // This would come from likelihood × prior in practice
    )
  )
  val posteriorDist = MVNormal(posteriorMu, posteriorCov)
  println(s"Target Posterior: p(θ|data) = MVNormal(μ=${posteriorMu}, Σ=...)")
  println("This represents our belief about parameters after seeing data")
  println("(In practice, this is intractable and what we approximate)")
  println()

  // 5. Define target posterior log probability function
  println("5. Setting up Posterior Log Probability")
  def posteriorLogProb(theta: Tensor1[ModelParam]): Tensor0 =
    posteriorDist.logpdf(theta)
  println("This function evaluates log p(θ|data) for any parameter value θ")
  println()

  // create  a composite flow
  val affineFlow1 = AffineFlow[LatentParam, LatentParam2]()
  val affineFlow2 = AffineFlow[LatentParam2, ModelParam]()

  case class CompositeFlowParams[P1, P2](
      flow1: P1,
      flow2: P2
  ) derives TensorTree,
        ToPyTree

  class TwoAffineFlows[A <: Label, B <: Label, C <: Label](
      af1: AffineFlow[A, B],
      af2: AffineFlow[B, C]
  ) extends Flow[A, C, CompositeFlowParams[AffineFlowParams[A, B], AffineFlowParams[B, C]]]:
    type Params = CompositeFlowParams[AffineFlowParams[A, B], AffineFlowParams[B, C]]

    def forwardSample(x: Tensor1[A])(params: Params): Tensor1[C] =
      val intermediate = af1.forwardSample(x)(params.flow1)
      af2.forwardSample(intermediate)(params.flow2)
  val compositeFlow = new TwoAffineFlows(affineFlow1, affineFlow2)

  println("6. Creating Normalizing Flow for Variational Inference")
  val nf = new NormalizingFlow(priorDist, compositeFlow)
  val flowLossFunction = nf.elbo(
    numSamples, // Number of samples for Monte Carlo estimation
    posteriorLogProb // Target log probability log p(θ|data)
  )

  println("The flow will learn f: z ~ p(z) → θ ≈ p(θ|data)")
  println("Maximizing: E[log p(f(z)|data) + log|det J_f(z)|]")
  println()

  // 7. Define loss function for gradient-based optimization
  def loss(
      params: CompositeFlowParams[
        AffineFlowParams[LatentParam, LatentParam2],
        AffineFlowParams[LatentParam2, ModelParam]
      ]
  ): Tensor0 =
    flowLossFunction(params) * Tensor0(-1f) // Negate to convert max → min problem

// Create parameters using the exact types the composite flow expects
  val initialParams = CompositeFlowParams(
    AffineFlowParams(
      weight = Tensor2[LatentParam, LatentParam2](
        Seq(
          Seq(1.0f, 0.0f), // 2x2 matrix to match LatentParam(2) dimensions
          Seq(0.0f, 1.0f)
        )
      ),
      bias = Tensor1[LatentParam2](Seq(0.0f, 0.0f))
    ),
    AffineFlowParams(
      weight = Tensor2[LatentParam2, ModelParam](
        Seq(
          Seq(1.0f, 0.0f), // 2x2 matrix to match LatentParam(2) dimensions
          Seq(0.0f, 1.0f)
        )
      ),
      bias = Tensor1[ModelParam](Seq(0.0f, 0.0f))
    )
  )

  val initialLoss = loss(initialParams)
  println(s"Initial parameters: $initialParams")
  println(s"Initial loss: ${initialLoss.toFloat}")
  println()

  // 9. Set up gradient computation
  println("8. Setting up Gradient Computation")
  val gradFn = Autodiff.grad(loss)

  // 10. Training loop - Variational inference optimization
  println("9. Variational Inference Training")
  val learningRate = 0.01f
  val numEpochs = 500

  val optimizer = GradientDescent(learningRate)

  println("Optimizing flow parameters to approximate p(θ|data)...")
  val finalParams = optimizer.optimize(gradFn, initialParams).take(numEpochs + 1).zipWithIndex.foldLeft(initialParams) {
    case (currentParams, (nextParams, iteration)) =>
      if iteration % 50 == 0 then
        val currentLoss = loss(nextParams)
        println(s"Epoch: $iteration, Negative Log-Likelihood: ${currentLoss.toFloat}")
        // println(s"  Flow Weight:\n${nextParams.weight}")
        // println(s"  Flow Bias: ${nextParams.bias}")
        println()
      nextParams
  }

  // 11. Final results
  println("10. Final Results")
  val finalLoss = loss(finalParams)
  println(s"Final negative log-likelihood: ${finalLoss.toFloat}")
  println()

  println("Learned Flow Parameters:")
  // println(s"Weight:\n${finalParams.weight}")
  // println(s"Bias: ${finalParams.bias}")
  println()

  println("True Model Parameters:")
  println(s"Weight:\n${trueWeight}")
  println(s"Bias: ${trueBias}")
  println()

  // 12. Evaluate quality of learned transformation
  println("11. Evaluation")
  // val weightError = (finalParams.weight - trueWeight).pow(Tensor0(2f)).sum
  // val biasError = (finalParams.bias - trueBias).pow(Tensor0(2f)).sum
  // val totalError = weightError + biasError

  // println(s"Weight MSE: ${weightError.toFloat}")
  // println(s"Bias MSE: ${biasError.toFloat}")
  // println(s"Total MSE: ${totalError.toFloat}")
  // println()

  // 13. Sample from learned posterior approximation
  println("12. Sampling from Approximate Posterior")
  val learnedFlow = AffineFlow[LatentParam, ModelParam]()
  val priorSamples = priorDist.sample[Sample](5) // Sample from prior p(z)
  // val posteriorSamples = learnedFlow.forward(priorSamples)(finalParams)  // Transform to θ ≈ p(θ|data)

  // println("Bayesian inference results:")
  // println(s"  Prior samples z ~ p(z) shape: ${priorSamples.shape}")
  // println(s"  Posterior samples θ = f(z) shape: ${posteriorSamples.shape}")
  // println(s"  Prior samples:\n${priorSamples}")
  // println(s"  Approximate posterior samples:\n${posteriorSamples}")

  println()
  println("Key features demonstrated:")
  println("- Bayesian posterior approximation using normalizing flows")
  println("- Variational inference with explicit prior p(z) and posterior p(θ|data)")
  println("- Case class parameters with TensorTree derivation")
  println("- Gradient-based optimization of flow parameters")
  println("- Proper change of variables with Jacobian determinant")
  println("- Transform samples: z ~ p(z) → θ ≈ p(θ|data)")
