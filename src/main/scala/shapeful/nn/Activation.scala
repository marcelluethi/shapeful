package shapeful.nn

import scala.language.experimental.namedTypeArguments
import scala.compiletime.summonInline
import shapeful.*
import shapeful.jax.Jax
import shapeful.tensor.TupleHelpers

object Activation:

  def relu[T <: Tuple](x: Tensor[T]) = x.relu
  def tanh[T <: Tuple](x: Tensor[T]) = x.tanh
  def sigmoid[T <: Tuple](x: Tensor[T]) = x.sigmoid
  inline def softmax[T <: Tuple, SoftmaxAxis <: Label](x: Tensor[T]) = x.softmax[SoftmaxAxis]
