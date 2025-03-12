package shapeful.autodiff

import shapeful.tensor.Tensor
import shapeful.tensor.Tensor.{Tensor0, Tensor1}
import shapeful.tensor.Dimension
import shapeful.tensor.Dimension.Symbolic




object Autodiff:

  def deriv[Shape <: Tuple](f : Function1[Tensor[Shape], Tensor[EmptyTuple.type]]) : Derivative[Tuple1[Tensor[Shape]]] =
    new Derivative1(f)

  def deriv[ShapeA <: Tuple, ShapeB <: Tuple](f : Function2[Tensor[ShapeA], Tensor[ShapeB], Tensor0]) : Derivative[(Tensor[ShapeA], Tensor[ShapeB])] =
    new Derivative2(f)

