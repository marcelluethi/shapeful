package shapeful.inference
import scala.language.experimental.namedTypeArguments
import shapeful.Label
import shapeful.*
import shapeful.distributions.MVNormal
import shapeful.autodiff.TensorTree
import shapeful.autodiff.Autodiff

trait Flow[From <: Label, To <: Label]:

  type Params

  def forwardSample(x: Tensor1[From])(params: Params): Tensor1[To]

  def forward[Sample <: Label](x: Tensor2[Sample, From])(params: Params): Tensor2[Sample, To] =
    x.vmap[VmapAxis = Sample](x => forwardSample(x)(params))

class AffineFlow[From <: Label, To <: Label] extends Flow[From, To]:

  import AffineFlow.AffineFlowParams

  type Params = AffineFlowParams[From, To]

  def forwardSample(x: Tensor1[From])(params: AffineFlowParams[From, To]): Tensor1[To] =
    val w = params.weight
    val b = params.bias
    x.matmul(w) + b

object AffineFlow:

  case class AffineFlowParams[From <: Label, To <: Label](
      weight: Tensor2[From, To],
      bias: Tensor1[To]
  ) derives TensorTree

def normalizingFlow[Samples <: Label, Latent <: Label, Output <: Label](
    priorDist: MVNormal[Latent],
    numSamples: Int,
    posteriorLogProb: Tensor1[Output] => Tensor0,
    flow: Flow[Latent, Output]
): flow.Params => Tensor0 =
  params =>
    val baseSamples = priorDist.sample[Samples](numSamples)

    // Transform all samples using the flow
    val transformedSamples = flow.forward(baseSamples)(params)

    // Compute log determinants using vmap over samples
    val logdet = baseSamples.vmap[VmapAxis = Samples] { sample =>
      // Define transformation function for a single sample using forwardSample
      def transformSample(x: Tensor1[Latent]): Tensor1[Output] =
        flow.forwardSample(x)(params)

      // Compute Jacobian for this sample using autodiff
      val jacFn = Autodiff.jacFwd(transformSample)
      val jac = jacFn(sample) // Jacobian matrix for this sample
      jac.det.abs.log
    }

    val baseLogProb = baseSamples.vmap[VmapAxis = Samples](priorDist.logpdf)
    val targetLogProb = transformedSamples.vmap[VmapAxis = Samples](posteriorLogProb)

    (baseLogProb + logdet + targetLogProb).mean

    // def transformSamples()
    // val transformedSamples = flow.forward(baseSamples)(params)
    // val jac = flow.jac(baseSamples)(params)
    // val logdet = jac.vmap[VmapAxis = Samples](t => t.det.abs.log)
    // val baseLogProb = baseSamples.vmap[VmapAxis = Samples](priorDist.logpdf)
    // val targetLogProb = transformedSamples.vmap[VmapAxis = Samples](posteriorLogProb)
    // (baseLogProb + logdet + targetLogProb).mean
