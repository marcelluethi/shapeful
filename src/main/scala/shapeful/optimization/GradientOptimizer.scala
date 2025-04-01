package shapeful.optimization

import shapeful.autodiff.Params
import shapeful.tensor.TensorOps
import shapeful.tensor.Tensor0
import shapeful.tensor.{Tensor, Variable}
import shapeful.tensor.TensorOps.sub
import shapeful.tensor.TensorOps.mul

class GradientOptimizer(lr : Float):

  def optimize(
    df: Params => Params,
    params: Params
  ) =
    Iterator.iterate(params) { currentState =>
        val grad = df(currentState)
        params.map((k, g) =>
            val g = grad.get[Variable](k)
            val p = currentState.get[Variable](k)
            val newp =  p.sub(g.mul(Tensor0(lr)))
            newp
        )
    }
