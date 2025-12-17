package nn

import shapeful.*
import shapeful.Conversions.given

case class GradientDescent[Params](df: Params => Params, lr: Float):
    def step(params: Params)(using paramTree: TensorTree[Params]) =
        val gradients = df(params)
        paramTree.zipMap(gradients, params, [T <: Tuple] => (n: Labels[T]) ?=> (g: Tensor[T], p: Tensor[T]) => 
            p - (g :* lr)
        )