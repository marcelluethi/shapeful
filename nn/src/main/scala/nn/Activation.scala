package nn

import shapeful.*
import shapeful.jax.Jax

object ActivationFunctions:

  // TODO rewrite relu, sigmoid to JAX
  
  def sigmoid[T <: Tuple : Labels, V : Value](t: Tensor[T, V]): Tensor[T, V] =
    val ones = Tensor.of[V].ones(t.shape)
    val minust = t.scale(Tensor0.of[V].apply(-1.0f))
    ones / (ones + (minust).exp)
  
  def relu[T <: Tuple : Labels, V : Value](t: Tensor[T, V]): Tensor[T, V] = 
    val zeros = Tensor.of[V].zeros(t.shape)
    maximum(t, zeros)

  def gelu[T <: Tuple : Labels, V : Value](t: Tensor[T, V]): Tensor[T, V] =
    Tensor.fromPy(Jax.jnn.gelu(t.jaxValue))

  def softmax[L: Label, V : Value](t: Tensor1[L, V]): Tensor1[L, V] =
    Tensor.fromPy(Jax.jnn.softmax(t.jaxValue, axis = 0))
