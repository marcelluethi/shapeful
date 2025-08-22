package shapeful.inference
import scala.language.experimental.namedTypeArguments
import shapeful.Label
import shapeful.*
import shapeful.distributions.MVNormal
import shapeful.autodiff.TensorTree
import shapeful.autodiff.Autodiff
import shapeful.autodiff.ToPyTree
import shapeful.tensor.TensorSlicing.splitAt
import shapeful.inference.flows.Flow

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
      posteriorLogProb: ModelParam => Tensor0,
      key: shapeful.random.Random.Key
  )(using fromTensor: FromTensor1[Output, ModelParam]): FlowParam => Tensor0 =

    type Samples = "Sample" // internal label for the samples

    params =>
      val baseSamples = baseDist.sample[Samples](numSamples, key)

      // Transform all samples using the flow
      val transformedSamples = flow.forward(baseSamples)(params)

      val logdet = baseSamples.vmap[VmapAxis = Samples] { sample =>
        flow.logDetJacobian(sample)(params)
      }

      val safeLogdet = logdet.clamp(1e-8f, 1e8f) // Prevent extreme values
      val baseLogProb = baseSamples.vmap[VmapAxis = Samples](baseDist.logpdf)

      val targetLogProb = transformedSamples.vmap[VmapAxis = Samples](t => posteriorLogProb(fromTensor.convert(t)))

      val logProbs = (targetLogProb - baseLogProb + safeLogdet)
      // val clampedLogProbs = logProbs.clamp(-1000f, 1000f) // Prevent extreme values
      // clampedLogProbs.mean
      logProbs.mean

  def generate[ModelParam](numSamples: Int, key: shapeful.random.Random.Key)(params: FlowParam)(using
      fromTensor: FromTensor1[Output, ModelParam]
  ): Seq[ModelParam] =
    type Samples = "Sample" // internal label for the samples
    val baseSamples = baseDist.sample[Samples](numSamples, key)
    val forwardedSamples = flow.forward(baseSamples)(params)
    forwardedSamples.unstack[VmapAxis = Samples].map(t => fromTensor.convert(t))
