package shapeful.autodiff

import shapeful.tensor.Tensor
import shapeful.tensor.Tensor.Tensor0
import shapeful.tensor.IsTensorTuple

trait Derivative[Tensors <: Tuple](using IsTensorTuple[Tensors]):
  def apply(params: Tensors): Tensors 

class Derivative1[Shape <: Tuple](f: Function1[Tensor[Shape], Tensor0]) extends Derivative[Tuple1[Tensor[Shape]]]:
  def apply(params: Tuple1[Tensor[Shape]]): Tuple1[Tensor[Shape]] = 
    val x = params.head
    val v = f(x)
    v.stensor.backward()
    Tuple1(x.grad())


class Derivative2[ShapeA <: Tuple, ShapeB <: Tuple](f: Function2[Tensor[ShapeA], Tensor[ShapeB], Tensor0]) extends Derivative[(Tensor[ShapeA], Tensor[ShapeB])]:
  def apply(params: (Tensor[ShapeA], Tensor[ShapeB])): (Tensor[ShapeA], Tensor[ShapeB]) = 
    val (x, y) = params
    val v = f(x, y)
    v.stensor.backward()
    (x.grad(), y.grad())
