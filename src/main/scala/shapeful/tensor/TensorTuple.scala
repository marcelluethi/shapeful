package shapeful.tensor

import shapeful.tensor.TensorOps.*

// Constraint to ensure each element in a tuple is a Tensor
trait IsTensorTuple[T <: Tuple]

object IsTensorTuple:
  // Base case: empty tuple is valid
  given IsTensorTuple[EmptyTuple] with {}

  // Inductive case: head must be a Tensor, recursively check tail
  given [H <: Tensor[?], Rest <: Tuple](using IsTensorTuple[Rest]): IsTensorTuple[H *: Rest] with {}


// Type class for updating tensor tuples
trait TensorTupleOps[T <: Tuple](using IsTensorTuple[T]):
  def update(params: T, gradients: T, lr: Tensor[EmptyTuple.type]): T

object TensorTupleOps:
  // Base case
  given TensorTupleOps[EmptyTuple] with
    def update(params: EmptyTuple, gradients: EmptyTuple, lr: Tensor[EmptyTuple.type]): EmptyTuple = EmptyTuple

  // Inductive case
  given [H <: Tensor[?], Rest <: Tuple](using IsTensorTuple[Rest], TensorTupleOps[Rest]): TensorTupleOps[H *: Rest] with
    def update(params: H *: Rest, gradients: H *: Rest, lr: Tensor[EmptyTuple.type]): H *: Rest =
      torch.noGrad{
        val param = params.head
        val grad = gradients.head
        val newHead = param.add(grad.mul(lr).asInstanceOf[param.type]).copy(requiresGrad = params.head.stensor.requiresGrad).asInstanceOf[H]
        val newTail = summon[TensorTupleOps[Rest]].update(params.tail, gradients.tail, lr)        
        newHead *: newTail
      }


