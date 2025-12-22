package nn

import shapeful.*
import shapeful.tensor.Value

case class GradientDescent[Params](df: Params => Params, lr: Float):
  def step(params: Params)(using paramTree: TensorTree[Params]) =
    val gradients = df(params)
    paramTree.zipMap(
      gradients,
      params,
      [T <: Tuple, V] =>
        (n: Labels[T], v: Value[V]) ?=>
          (g: Tensor[T, V], p: Tensor[T, V]) =>
            val lrTensor = Tensor0(lr)(using v)
            p - (g :* lrTensor)
    )
