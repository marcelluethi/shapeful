package nn

import shapeful.*
import shapeful.tensor.Value

object ActivationFunctions:

  def sigmoid[T <: Tuple: Labels, V: Value](t: Tensor[T, V]): Tensor[T, V] =
    val ones = Tensor.of[V].ones(t.shape)
    val minusOne = Tensor0(-1.0f)(using summon[Value[V]])
    val minust = t :* minusOne
    ones / (ones + (minust).exp)

  def relu[T <: Tuple: Labels, V: Value](t: Tensor[T, V]): Tensor[T, V] =
    val zeros = Tensor.of[V].zeros(t.shape)
    maximum(t, zeros)
