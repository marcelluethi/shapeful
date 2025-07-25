package shapeful.inference.flows

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.autodiff.Autodiff

trait Flow[From <: Label, To <: Label, P]:

  type Params = P

  def forwardSample(x: Tensor1[From])(params: Params): Tensor1[To]

  def forward[Sample <: Label](x: Tensor2[Sample, From])(params: Params): Tensor2[Sample, To] =
    x.vmap[VmapAxis = Sample](x => forwardSample(x)(params))

  // Optional analytical log determinant - override for efficiency
  def logDetJacobian(x: Tensor1[From])(params: Params): Tensor0 =
    // Default: use autodiff (slower but general)
    def transformSample(x: Tensor1[From]): Tensor1[To] = forwardSample(x)(params)
    val jacFn = Autodiff.jacFwd(transformSample)
    val jac = jacFn(x)
    val logdet = jac.det.abs.log

    // cap to limit numerical instabilities
    logdet.clamp(-10.0f, 10.0f)
