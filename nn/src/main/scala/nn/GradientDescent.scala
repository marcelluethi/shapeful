package nn

import shapeful.*
import me.shadaj.scalapy.readwrite.Writer

case class GradientDescent[Params](df: Params => Params, lr: Float)(
  using writer: Writer[Float]
):

  val lrTensor: Tensor0[Float] = Tensor0(TensorValue[Float]).const(lr)

  def step(params: Params)(using paramTree: TensorTree[Params]) =
    val gradients = df(params)
    paramTree.zipMap(
      gradients,
      params,
      [T <: Tuple, V2] =>
        (tv: TensorValue[V2]) =>
        (n: Labels[T]) ?=>
        (g: Tensor[T, V2], p: Tensor[T, V2]) =>
          val lrTensorV = lrTensor.asValue(tv)
          p - (g :* lrTensorV)
    )
