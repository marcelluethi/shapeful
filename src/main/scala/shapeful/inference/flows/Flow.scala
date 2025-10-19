package shapeful.inference.flows

import shapeful.*
import shapeful.autodiff.Autodiff

trait Flow[From <: Label, To <: Label, P]:

  type Params = P

  def forwardSample(x: Tensor1[From])(params: Params): Tensor1[To]

  inline def forward[Sample <: Label](x: Tensor2[Sample, From])(params: Params): Tensor2[Sample, To] =
    x.vmap(Axis[Sample]) { x => forwardSample(x)(params) }

  // Optional analytical log determinant - override for efficiency
  def logDetJacobian(x: Tensor1[From])(params: Params): Tensor0 =
    // Default: use autodiff (slower but general)
    def transformSample(x: Tensor1[From]): Tensor1[To] = forwardSample(x)(params)
    val jacFn = Autodiff.jacFwd(transformSample)
    val jac = jacFn(x)
    val logdet = jac.det.abs.log

    val dim = x.shape.dim[From]
    val maxReasonableLogDet = math.log(1e8).toFloat * dim
    val minReasonableLogDet = math.log(1e-8).toFloat * dim

    logdet.clamp(minReasonableLogDet, maxReasonableLogDet)
