package shapeful.autodiff

import shapeful.tensor.{Tensor0, Tensor, Variable}
import torch.Float32


/** Parameters are stored in a tree structure, where each node can have children
  * of different types The param tree as such is jus represented as tensor, but
  * when reading we specify the type to get the correct type of the parameter
  *
  * The idea is similar to jax pytrees and json structures
  */
class Params(m: Map[String, Variable]):
  def get[A <: Variable](key: String): A = m(key).asInstanceOf[A]

  def update[A <: Variable](pair : (String, Variable)): Params = 
    val (k, v) = pair
    if m.contains(k) then
      Params(m.updated(k, v))
    else
      throw new NoSuchElementException(s"Key $k not found in Params")

  def map(f: Function2[String, Variable, Variable]): Params =
    torch.noGrad {
      Params(m.map { case (key, value) => (key, f(key, value)) })
    }

    

def deriv(f: Params => Tensor0[Float32]): Params => Params =
  p =>
    val value = f(p)
    value.repr.backward()
    p.map((k, v) => Variable.fromTorch(v.repr.grad.get.requiresGrad = false))
