package shapeful.optimization

import shapeful.tensor.Tensor.Tensor0

import shapeful.tensor.Tensor.Tensor1
import shapeful.tensor.Tensor
import shapeful.autodiff.Derivative
import shapeful.tensor.TensorTupleOps
import shapeful.tensor.IsTensorTuple


class GradientDescent(lr_ : Float):
  val lr = Tensor[EmptyTuple.type](lr_, requiresGrad = false)
  
  def optimize[Tensors <: Tuple : IsTensorTuple](
    df: Derivative[Tensors], 
    init: Tensors
  )(using ops: TensorTupleOps[Tensors]): Iterator[Tensors] =
    Iterator.iterate(init) { params =>
      val gradients = df(params)
      ops.update(params, gradients, lr)
    }

