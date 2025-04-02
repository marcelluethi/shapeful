package shapeful.autodiff

import shapeful.tensor.{Tensor0, Tensor, Variable}
import torch.Float32

trait TensorTree[T <: Tensor[Float32], TT <: TensorTree[T, TT]](m: Map[String, T]):

  def get[A <: T](key: String): A = m(key).asInstanceOf[A]
  def update[A <: T](pair : (String, A)): TT 

class Grads(m: Map[String, Tensor[Float32]]) extends TensorTree[Tensor[Float32], Grads](m):


  def update[A <: Tensor[Float32]](pair : (String, A)): Grads = 
    torch.noGrad {
      val (k, v) = pair
      if m.contains(k) then
        Grads(m.updated(k, v))
      else
        throw new NoSuchElementException(s"Key $k not found in Params")
    }

  def map(f: Function2[String, Tensor[Float32], Tensor[Float32]]): Grads =
    torch.noGrad {
      Grads(m.map { case (key, value) => (key, f(key, value)) })
    }

class Params(m: Map[String, Variable]) extends TensorTree[Variable, Params](m):
  def update[A <: Variable](pair : (String, A)): Params = 
    val (k, v) = pair
    if m.contains(k) then
      Params(m.updated(k, v))
    else
      throw new NoSuchElementException(s"Key $k not found in Params")

  def map(f: Function2[String, Variable, Variable]): Params =
      Params(m.map { case (key, value) => (key, f(key, value)) })
  
  /**
   * Tracking of gradients in function f is disabled
   */
  def mapToGrad(f: Function2[String, Variable, Tensor[Float32]]): Grads =
    torch.noGrad {
      Grads(m.map { case (key, value) => (key, f(key, value)) })   
    }

// /** Parameters are stored in a tree structure, where each node can have children
//   * of different types The param tree as such is jus represented as tensor, but
//   * when reading we specify the type to get the correct type of the parameter
//   *
//   * The idea is similar to jax pytrees and json structures
//   */
// class Params(m: Map[String, Variable]) extends TensorTree(m)
    

def deriv(f: Params => Tensor0[Float32]): Params => Grads =
  p =>
    val value = f(p)
    value.repr.backward()
    p.mapToGrad((k, v) => Tensor.fromTorch(v.repr.grad.get.requiresGrad = false))
