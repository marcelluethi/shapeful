package shapeful.linalg

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.jax.Jax

object Linalg:

  /**
   * Computes the inverse of a square matrix.
   * 
   * @param t A square Tensor2 matrix
   * @return The matrix inverse as a Tensor2 with transposed dimensions
   */
  def inverse[A <: Label, B <: Label](
        t: Tensor2[A, B]
    ) : Tensor2[B, A] = 
      // Call JAX's linalg.inv function
      val invMatrix = Jax.jnp.linalg.inv(t.jaxValue)
      
      // Create a new tensor with the appropriate shape (dimensions are swapped)
      val rowDim = t.shape.dims(1) // B becomes first dimension
      val colDim = t.shape.dims(0) // A becomes second dimension
      val resultShape = Shape2[B, A](rowDim, colDim)
      
      new Tensor2[B, A](resultShape, invMatrix.as[Jax.PyDynamic], t.dtype)

  /**
   * Computes the Cholesky decomposition of a symmetric positive-definite matrix.
   * The Cholesky decomposition is a matrix L such that t = L * L.T
   * 
   * @param t A symmetric positive-definite Tensor2 matrix
   * @return The lower-triangular Cholesky factor as a Tensor2
   */
  def cholesky[A <: Label, B <: Label](
        t: Tensor2[A, B]
    ): Tensor2[A, B] = 
      // Call JAX's linalg.cholesky function
      val cholMatrix = Jax.jnp.linalg.cholesky(t.jaxValue)
      
      // Create a new tensor with the same shape
      new Tensor2[A, B](t.shape, cholMatrix.as[Jax.PyDynamic], t.dtype)
