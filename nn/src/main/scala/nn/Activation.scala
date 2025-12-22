package nn

import shapeful.*
import shapeful.jax.Jax
import shapeful.Conversions.given

object ActivationFunctions:

  // TODO rewrite relu, sigmoid to JAX
  
  def sigmoid[T <: Tuple : Labels](t: Tensor[T]): Tensor[T] =
    val ones = Tensor.ones(t.shape)
    val minust = t :* -1.0f
    ones / (ones + (minust).exp)
  
  def relu[T <: Tuple : Labels](t: Tensor[T]): Tensor[T] = 
    val zeros = Tensor.zeros(t.shape)
    maximum(t, zeros)

  def gelu[T <: Tuple : Labels](t: Tensor[T]): Tensor[T] =
    Tensor.fromPy(Jax.jnn.gelu(t.jaxValue))

  def softmax[L: Label](t: Tensor1[L]): Tensor1[L] =
    Tensor.fromPy(Jax.jnn.softmax(t.jaxValue, axis = 0))
