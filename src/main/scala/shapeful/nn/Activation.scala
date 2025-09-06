package shapeful.nn

import scala.language.experimental.namedTypeArguments
import scala.compiletime.summonInline
import shapeful.*
import shapeful.jax.Jax
import shapeful.tensor.TupleHelpers

trait Activation[T <: Tuple] extends Function1[Tensor[T], Tensor[T]]:
  def apply(x: Tensor[T]): Tensor[T]

object Activation:

  case class ReLu[T <: Tuple]() extends Activation[T]:
    def apply(x: Tensor[T]): Tensor[T] = x.relu

  case class Tanh[T <: Tuple]() extends Activation[T]:
    def apply(x: Tensor[T]): Tensor[T] = x.tanh

  case class Sigmoid[T <: Tuple]() extends Activation[T]:
    def apply(x: Tensor[T]): Tensor[T] = x.sigmoid

  case class Softmax[T <: Tuple]():
    inline def apply[SoftmaxAxis <: Label](t: Tensor[T]): Tensor[T] =
      t.softmax[SoftmaxAxis]
