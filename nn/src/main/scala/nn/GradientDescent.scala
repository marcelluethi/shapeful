package nn

import shapeful.*

case class GradientDescent[Params](df: Params => Params, lr: Float):
    def step(params: Params)(using paramTree: TensorTree[Params]) =
        val gradients = df(params)
        paramTree.zipMap(gradients, params, [T <: Tuple, V] => (n: Labels[T], v: Value[V]) ?=> (g: Tensor[T, V], p: Tensor[T, V]) => 
            p - g.scale(Tensor0.of[V].apply(lr))
        )