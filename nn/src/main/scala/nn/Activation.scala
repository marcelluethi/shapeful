package nn

import shapeful.*

object ActivationFunctions:

  def sigmoid[T <: Tuple: Labels, V](t: Tensor[T, V]): Tensor[T, V] =
    val ones = Tensor(t.tv).ones(t.shape)
    val minusOne = -Tensor0(t.tv).one
    val minust = t :* minusOne
    ones / (ones + (minust).exp)

  def relu[T <: Tuple: Labels, V](t: Tensor[T, V]): Tensor[T, V] =
    val zeros = Tensor(t.tv).zeros(t.shape)
    maximum(t, zeros)
