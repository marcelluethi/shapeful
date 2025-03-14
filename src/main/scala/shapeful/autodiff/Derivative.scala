package shapeful.autodiff

import shapeful.tensor.Tensor
import shapeful.tensor.Tensor.Tensor0
import shapeful.tensor.IsTensorTuple

trait Derivative[Tensors <: Tuple](using IsTensorTuple[Tensors]):
  def apply(params: Tensors): Tensors 

class Derivative1[Dims <: Tuple](f: Function1[Tensor[Dims], Tensor0]) extends Derivative[Tuple1[Tensor[Dims]]]:
  def apply(params: Tuple1[Tensor[Dims]]): Tuple1[Tensor[Dims]] = 
    val x = params.head
    val v = f(x)
    v.stensor.backward()
    Tuple1(x.grad())


class Derivative2[DimsA <: Tuple, DimsB <: Tuple](f: Function2[Tensor[DimsA], Tensor[DimsB], Tensor0]) extends Derivative[(Tensor[DimsA], Tensor[DimsB])]:
  def apply(params: (Tensor[DimsA], Tensor[DimsB])): (Tensor[DimsA], Tensor[DimsB]) = 
    val (x, y) = params
    val v = f(x, y)
    v.stensor.backward()
    (x.grad(), y.grad())

class Derivative3[DimsA <: Tuple, DimsB <: Tuple, DimsC <: Tuple](f: Function3[Tensor[DimsA], Tensor[DimsB], Tensor[DimsC], Tensor0]) extends Derivative[(Tensor[DimsA], Tensor[DimsB], Tensor[DimsC])]:
  def apply(params: (Tensor[DimsA], Tensor[DimsB], Tensor[DimsC])): (Tensor[DimsA], Tensor[DimsB], Tensor[DimsC]) = 
    val (x, y, z) = params
    val v = f(x, y, z)
    v.stensor.backward()
    (x.grad(), y.grad(), z.grad())


class Derivative4[DimsA <: Tuple, DimsB <: Tuple, DimsC <: Tuple, DimsD <: Tuple](f: Function4[Tensor[DimsA], Tensor[DimsB], Tensor[DimsC], Tensor[DimsD], Tensor0]) extends Derivative[(Tensor[DimsA], Tensor[DimsB], Tensor[DimsC], Tensor[DimsD])]:
  def apply(params: (Tensor[DimsA], Tensor[DimsB], Tensor[DimsC], Tensor[DimsD])): (Tensor[DimsA], Tensor[DimsB], Tensor[DimsC], Tensor[DimsD]) = 
    val (p1, p2, p3, p4) = params
    val v = f(p1, p2, p3, p4)
    v.stensor.backward()
    (p1.grad(), p2.grad(), p3.grad(), p4.grad())