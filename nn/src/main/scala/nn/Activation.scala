package nn

import shapeful.*
import shapeful.Conversions.given

object ActivationFunctions:

  def sigmoid[T <: Tuple : Labels](t: Tensor[T]): Tensor[T] =
    val ones = Tensor.ones(t.shape)
    val minust = t :* -1.0f
    ones / (ones + (minust).exp)
  
  def relu[T <: Tuple : Labels](t: Tensor[T]): Tensor[T] = 
    val zeros = Tensor.zeros(t.shape)
    maximum(t, zeros)