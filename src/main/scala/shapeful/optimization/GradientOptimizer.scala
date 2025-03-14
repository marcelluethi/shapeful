package shapeful.optimization

import shapeful.tensor.Tensor.Tensor0

import shapeful.tensor.Tensor.Tensor1
import shapeful.tensor.Tensor
import shapeful.autodiff.Derivative
import shapeful.tensor.TensorTupleOps
import shapeful.tensor.IsTensorTuple
import scala.collection.AbstractIterator


class GradientOptimizer(lr_ : Float):
  val lr = Tensor[EmptyTuple.type](lr_, requiresGrad = false)
  
  def optimize[Tensors <: Tuple : IsTensorTuple](
    df: Derivative[Tensors], 
    init: Tensors
  )(using ops: TensorTupleOps[Tensors]): Iterator[Tensors] =
    Iterator.iterate(init) { currentState =>
        val gradients = df(currentState)
        torch.noGrad {
          val result = currentState
          ops.update(currentState, gradients, lr)
        }
      }
    

  
