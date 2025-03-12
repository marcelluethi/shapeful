package shapeful.optimization

import shapeful.tensor.Tensor.Tensor0

import shapeful.tensor.Tensor.Tensor1
import shapeful.tensor.Tensor
import shapeful.tensor.multScalar
import shapeful.tensor.add
import shapeful.autodiff.Derivative
import shapeful.autodiff.TensorTupleOps


class GradientDescent(lr_ : Float):
  val lr = Tensor[EmptyTuple.type](lr_, requiresGrad = false)
  
  def optimize[T <: Tuple](
    df: Derivative[T], 
    init: T
  )(using ops: TensorTupleOps[T]): Iterator[T] =
    Iterator.iterate(init) { params =>
      val gradients = df(params)
      ops.update(params, gradients, lr)
    }

    // (0 until steps).foldLeft(init) { case (params, i) =>
    //   println(s"Iteration $i: ${params}")
      
    //   val gradients = df(params)
    //   ops.update(params, gradients, lr)
    //}

