package shapeful.autodiff

import shapeful.tensor.{Tensor0, Tensor, Variable}
import torch.Float32
import shapeful.tensor.{Tensor1, Tensor2, Tensor3}

import shapeful.tensor.Shape
import shapeful.tensor.Shape2

trait TensorTree[T <: Tensor[Float32], TT <: TensorTree[T, TT]](m: Map[String, T]):

  def get[A <: T](key: String): A = m(key).asInstanceOf[A]
  def update[A <: T](pair : (String, A)): TT 
  def values : Iterable[T] = m.values

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
  def foreach(f: Function2[String, Variable, Unit]): Unit =
    m.foreach { case (key, value) => f(key, value) }
    
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
    
object Autodiff:
  def deriv(f: Params => Tensor0[Float32]): Params => Grads =
    p =>
      val value = f(p)
      value.repr.backward()
      p.mapToGrad((k, v) => Tensor.fromTorch(v.repr.grad.get.requiresGrad = false))

  def jacobian[A <: Singleton](f : Params => Tensor1[A, Float32]): Params => Seq[Grads] = ???

  def jacobianMatrix[Sample <: Singleton, A <: Singleton, B <: Singleton](f: Params => Tensor2[Sample, A, Float32]): 
    Params => Tensor3[Sample, A, B, Float32] = ???
    
  // def jacobian[A <: Singleton, B <: Singleton](f: Params => Tensor1[A, Float32]): Params => Seq[Grads] = {
  //   p => 

  //   val output = f(p)
  //   val outputSize = output.shape.dim1 // Assuming Tensor.shape.product gives total elements
  //   val inputParams = p.values.toArray // Assuming Params.values gives an Iterable of tensors
  //   val inputSize = inputParams.map(_.shape.dims.product).sum
  //   val jacobianMatrix = Array.ofDim[Float](outputSize, inputSize)
    
  //   for (i <- 0 until outputSize) {
  //     val output = f(p)
  //     val outputElement = output.repr.reshape(outputSize)(i) // get each element of the output tensor
  //     outputElement.backward()
      
  //     var inputIndex = 0
  //     for (param <- inputParams) {
  //       val paramSize = param.shape.dims.product
  //       val paramGrad = param.repr.grad.get.reshape(paramSize).toArray // Flatten gradient to array

  //       for (j <- 0 until paramSize) {
  //         jacobianMatrix(i)(inputIndex + j) = paramGrad(j)
  //       }
  //       inputIndex += paramSize
  //       param.repr.grad.forall(g => {g.zero_(); true})// Reset gradients for next iteration
  //     }
  //   }
  //   val jacMatrixSeq = jacobianMatrix.flatten.toSeq // Flatten the matrix to a sequence
  //   new Tensor2(new Shape2[A, B](outputSize, inputSize), torch.Tensor(jacMatrixSeq).reshape(outputSize, inputSize)) // Create a Tensor from the resulting matrix
  // }

  //   def jacobian2[Data <: Singleton, A <: Singleton, B <: Singleton](f: Params => Tensor2[Data, A, Float32]): Params => Tensor3[Data, A, B, Float32] = {
  //   p => 

  //   // val output = f(p)
  //   // val outputSize = output.shape.dim1 // Assuming Tensor.shape.product gives total elements
  //   // val inputParams = p.values.toArray // Assuming Params.values gives an Iterable of tensors
  //   // val inputSize = inputParams.map(_.shape.dims.product).sum
  //   // val jacobianMatrix = Array.ofDim[Float](outputSize, inputSize)
    
  //   // for (i <- 0 until outputSize) {
  //   //   val output = f(p)
  //   //   val outputElement = output.repr.reshape(outputSize)(i) // get each element of the output tensor
  //   //   outputElement.backward()
      
  //   //   var inputIndex = 0
  //   //   for (param <- inputParams) {
  //   //     val paramSize = param.shape.dims.product
  //   //     val paramGrad = param.repr.grad.get.reshape(paramSize).toArray // Flatten gradient to array

  //   //     for (j <- 0 until paramSize) {
  //   //       jacobianMatrix(i)(inputIndex + j) = paramGrad(j)
  //   //     }
  //   //     inputIndex += paramSize
  //   //     param.repr.grad.forall(g => {g.zero_(); true})// Reset gradients for next iteration
  //   //   }
  //   // }
  //   // val jacMatrixSeq = jacobianMatrix.flatten.toSeq // Flatten the matrix to a sequence
  //   // new Tensor2(new Shape2[A, B](outputSize, inputSize), torch.Tensor(jacMatrixSeq).reshape(outputSize, inputSize)) // Create a Tensor from the resulting matrix
  //   ???
  // }