package shapeful.autodiff

import shapeful.tensor.Tensor
import shapeful.tensor.Tensor.{Tensor0, Tensor1}
import shapeful.tensor.Dimension
import shapeful.tensor.Dimension.Symbolic




object Autodiff:

  def deriv[Dims <: Tuple](f : Function1[Tensor[Dims], Tensor[EmptyTuple.type]]) : Derivative[Tuple1[Tensor[Dims]]] =
    new Derivative1(f)

  def deriv[DimsA <: Tuple, DimsB <: Tuple](f : Function2[Tensor[DimsA], Tensor[DimsB], Tensor0]) : Derivative[(Tensor[DimsA], Tensor[DimsB])] =
    new Derivative2(f)

