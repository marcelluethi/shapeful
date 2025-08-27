package shapeful.nn

import scala.language.experimental.namedTypeArguments

import shapeful.*

trait Activation[T <: Tuple] extends Function1[Tensor[T], Tensor[T]]:
  def apply(x: Tensor[T]): Tensor[T]

object Activation:

  case class ReLu[T <: Tuple]() extends Activation[T]:
    def apply(x: Tensor[T]): Tensor[T] = x.relu

  case class Sigmoid[T <: Tuple]() extends Activation[T]:
    def apply(x: Tensor[T]): Tensor[T] = x.sigmoid
