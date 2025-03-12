package shapeful.autodiff

import shapeful.tensor.Tensor
import shapeful.tensor.Tensor.{Tensor0, Tensor1}
import shapeful.tensor.Dimension
import shapeful.tensor.Dimension.Symbolic
import shapeful.tensor.{add, mult}
import spire.math.Empty
import shapeful.tensor.multScalar

// Constraint to ensure each element in a tuple is a Tensor
trait IsTensorTuple[T <: Tuple]

// Base case: empty tuple is valid
given IsTensorTuple[EmptyTuple] with {}

// Inductive case: head must be a Tensor, recursively check tail
given [H <: Tensor[_], Rest <: Tuple](using IsTensorTuple[Rest]): IsTensorTuple[H *: Rest] with {}

trait Derivative[T <: Tuple](using IsTensorTuple[T]):
  def apply(params: T): T 

class Derivative1[A <: Tuple](f: Function1[Tensor[A], Tensor[EmptyTuple.type]]) extends Derivative[Tuple1[Tensor[A]]]:
  def apply(params: Tuple1[Tensor[A]]): Tuple1[Tensor[A]] = 
    val x = params.head
    val v = f(x)
    v.stensor.backward()
    Tuple1(x.grad())


class Derivative2[A <: Tuple, B <: Tuple](f: Function2[Tensor[A], Tensor[B], Tensor[EmptyTuple.type]]) extends Derivative[(Tensor[A], Tensor[B])]:
  def apply(params: (Tensor[A], Tensor[B])): (Tensor[A], Tensor[B]) = 
    val (x, y) = params
    val v = f(x, y)
    v.stensor.backward()
    (x.grad(), y.grad())

object Autodiff:

  def deriv[A <: Tuple](f : Function1[Tensor[A], Tensor[EmptyTuple.type]]) : Derivative[Tuple1[Tensor[A]]] =
    new Derivative1(f)

  def deriv[A <: Tuple, B <: Tuple](f : Function2[Tensor[A], Tensor[B], Tensor[EmptyTuple.type]]) : Derivative[(Tensor[A], Tensor[B])] =
    new Derivative2(f)


// Type class for updating tensor tuples
trait TensorTupleOps[T <: Tuple]:
  def update(params: T, gradients: T, lr: Tensor[EmptyTuple.type]): T

object TensorTupleOps:
  // Base case
  given TensorTupleOps[EmptyTuple] with
    def update(params: EmptyTuple, gradients: EmptyTuple, lr: Tensor[EmptyTuple.type]): EmptyTuple = EmptyTuple

  // Inductive case
  given [H <: Tensor[_], Rest <: Tuple](using TensorTupleOps[Rest]): TensorTupleOps[H *: Rest] with
    def update(params: H *: Rest, gradients: H *: Rest, lr: Tensor[EmptyTuple.type]): H *: Rest =
      val param = params.head
      val grad = gradients.head
      val newHead = param.add(grad.multScalar(lr).asInstanceOf[param.type]).copy(requiresGrad = true).asInstanceOf[H]
      //val newHead = params.head.copy(requiresGrad = true).asInstanceOf[H] 
      val newTail = summon[TensorTupleOps[Rest]].update(params.tail, gradients.tail, lr)
      newHead *: newTail

