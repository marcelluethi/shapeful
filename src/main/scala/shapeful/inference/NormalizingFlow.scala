package shapeful.inference
import scala.language.experimental.namedTypeArguments
import shapeful.Label
import shapeful.*
import shapeful.distributions.MVNormal
import shapeful.autodiff.TensorTree
import shapeful.autodiff.Autodiff
import NormalizingFlow.Flow
import shapeful.autodiff.ToPyTree
import shapeful.inference.AffineFlow.AffineFlowParams

trait FromTensor1[L <: Label, Param]:
  def convert(tensor: Tensor1[L]): Param

object FromTensor1:
  given [L <: Label]: FromTensor1[L, Tensor1[L]] with
    def convert(tensor: Tensor1[L]): Tensor1[L] = tensor.relabel[Tuple1[L]]

class NormalizingFlow[Latent <: Label, Output <: Label, FlowParam](
    val baseDist: MVNormal[Latent],
    val flow: Flow[Latent, Output, FlowParam]
):

  def elbo[ModelParam](
      numSamples: Int,
      posteriorLogProb: ModelParam => Tensor0
  )(using fromTensor: FromTensor1[Output, ModelParam]): FlowParam => Tensor0 =

    type Samples = "Sample" // internal label for the samples

    params =>
      val baseSamples = baseDist.sample[Samples](numSamples)

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

      val baseLogProb = baseSamples.vmap[VmapAxis = Samples](baseDist.logpdf)

      val targetLogProb = transformedSamples.vmap[VmapAxis = Samples](t => posteriorLogProb(fromTensor.convert(t)))

      (targetLogProb - baseLogProb + logdet).mean

  def generate[ModelParam](numSamples: Int)(params: FlowParam)(using
      fromTensor: FromTensor1[Output, ModelParam]
  ): Seq[ModelParam] =
    type Samples = "Sample" // internal label for the samples
    val baseSamples = baseDist.sample[Samples](numSamples)
    val forwardedSamples = flow.forward(baseSamples)(params)
    forwardedSamples.split[VmapAxis = Samples].map(t => fromTensor.convert(t))

object NormalizingFlow:

  trait Flow[From <: Label, To <: Label, P]:

    type Params = P

    def forwardSample(x: Tensor1[From])(params: Params): Tensor1[To]

    def forward[Sample <: Label](x: Tensor2[Sample, From])(params: Params): Tensor2[Sample, To] =
      x.vmap[VmapAxis = Sample](x => forwardSample(x)(params))

class AffineFlow[From <: Label, To <: Label] extends Flow[From, To, AffineFlow.AffineFlowParams[From, To]]:

  type Params = AffineFlowParams[From, To]

  def forwardSample(x: Tensor1[From])(params: AffineFlowParams[From, To]): Tensor1[To] =
    val w = params.weight
    val b = params.bias
    x.matmul(w) + b

object AffineFlow:

  case class AffineFlowParams[From <: Label, To <: Label](
      weight: Tensor2[From, To],
      bias: Tensor1[To]
  ) derives TensorTree,
        ToPyTree
