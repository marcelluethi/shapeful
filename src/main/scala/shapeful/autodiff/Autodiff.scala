package shapeful.autodiff

import shapeful.tensor.Tensor
import shapeful.tensor.Tensor.{Tensor0, Tensor1}
import shapeful.tensor.Dimension
import shapeful.tensor.Dimension.Symbolic
import shapeful.tensor.{add, mult}
import spire.math.Empty

// Constraint to ensure each element in a tuple is a Tensor
trait IsTensorTuple[T <: Tuple]

// Base case: empty tuple is valid
given IsTensorTuple[EmptyTuple] with {}

// Inductive case: head must be a Tensor, recursively check tail
given [H <: Tensor[_], Rest <: Tuple](using IsTensorTuple[Rest]): IsTensorTuple[H *: Rest] with {}

// The differentiable function with constraint
trait DifferentiableFunction[T <: Tuple](using IsTensorTuple[T]):
  def apply(params: T): Tensor[EmptyTuple.type]
  def deriv(params: T): T 


class DifferentiableFunction2[A <: Tuple, B <: Tuple](f: Function2[Tensor[A], Tensor[B], Tensor[EmptyTuple.type]]) extends DifferentiableFunction[(Tensor[A], Tensor[B])]:
      def apply(xy  : (Tensor[A], Tensor[B])) : Tensor[EmptyTuple.type] = 
        // val (x,y) = xy
        // f(x, y)
        ???
      def deriv(xy : (Tensor[A], Tensor[B])): (Tensor[A], Tensor[B]) =
        //(xy._1, xy._2)
        val (x, y) = xy
        val v = f(x, y)
        v.stensor.backward()
   
        (x.grad(), y.grad())
      

// Type class for updating tensor tuples
trait TensorTupleOps[T <: Tuple]:
  def update(params: T, gradients: T, lr: Tensor[EmptyTuple.type]): T

// Base case
given TensorTupleOps[EmptyTuple] with
  def update(params: EmptyTuple, gradients: EmptyTuple, lr: Tensor[EmptyTuple.type]): EmptyTuple = EmptyTuple

// Inductive case
given [H <: Tensor[_], Rest <: Tuple](using TensorTupleOps[Rest]): TensorTupleOps[H *: Rest] with
  def update(params: H *: Rest, gradients: H *: Rest, lr: Tensor[EmptyTuple.type]): H *: Rest =
    val newHead = params.head.copy(requiresGrad = true).asInstanceOf[H] 
    val newTail = summon[TensorTupleOps[Rest]].update(params.tail, gradients.tail, lr)
    newHead *: newTail

class GeneralGradientDescent(lr_ : Float, steps : Int):
  val lr = Tensor[EmptyTuple.type](lr_, requiresGrad = false)
  
  def optimize[T <: Tuple](
    f: DifferentiableFunction[T], 
    init: T
  )(using ops: TensorTupleOps[T]): T =
    (0 until steps).foldLeft(init) { case (params, i) =>
      println(s"Iteration $i: ${params}")
      
      val gradients = f.deriv(params)
      ops.update(params, gradients, lr)
    }


