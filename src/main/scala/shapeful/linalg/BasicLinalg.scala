package shapeful.linalg

import shapeful.tensor.Tensor2
import torch.Float32
import shapeful.tensor.TensorOps.norm
import shapeful.tensor.Tensor0
import shapeful.tensor.TensorOps.div
import shapeful.tensor.Tensor2Ops.matmul
import shapeful.tensor.Shape
import shapeful.tensor.->>
import shapeful.tensor.TensorOps.mul
import shapeful.tensor.TensorOps.sub
import shapeful.tensor.Shape2
import shapeful.tensor.FromRepr
import shapeful.tensor.TensorOps.add
import shapeful.tensor.TensorOps.sqrt
import shapeful.tensor.TensorOps.abs

object BasicLinalg:

    /**
      * 
      Compute the inverse of a matrix using Newton's method.
    
    Args:
        A: Square tensor to invert
        num_iterations: Number of Newton iterations
        initial_guess: Initial guess for inverse (defaults to scaled identity)
        
    Returns:
        Approximate inverse of A
        */

    def inverse[A <: Singleton, B <: Singleton](
        t: Tensor2[A, B, Float32],  numIterations : Int = 10, initialGuess: Option[Tensor2[B, A, Float32]] = None
    )(using fromReprAA : FromRepr[Float32, Tensor2[A, A, Float32]]): Tensor2[B, A, Float32] =

        // require(t.shape.dim1 == t.shape.dim2, "Tensor must be square to compute inverse")

        // val n = t.shape.dim1
        // // Initial guess: X_0 = alpha * I, where alpha = 1/||A||_F
        // val X : Tensor2[B, A, Float32]= initialGuess match
        //     case Some(guess) => guess
        //     case None =>
        //         val alpha = Tensor0(1.0f).div(t.norm)
        //         Tensor2.eye(new Shape2[B, A](n, n)).mul(alpha)

        // val I = Tensor2.eye(new Shape2[A, A](n, n))
        // // Newton iterations: X_{k+1} = X_k(2I - AX_k)
        // var Xk = X
        // var prevError = Float.MaxValue
        // for (i <- 0 until numIterations) {
        //     val nextX = Xk.matmul(I.mul(Tensor0(2.0f)).sub(t.matmul(Xk)))
        //     val error = (t.matmul(nextX).sub(I)).norm // Residual error ||AX - I||
            
        //     if (error.item < 1e-6f) {
        //         return nextX
        //     }
            
        //     prevError = error.item
        //     Xk = nextX
        // }
        // Xk
        new Tensor2[B, A, Float32](new Shape2[B, A](t.shape.dim2, t.shape.dim1), torch.linalg.BasicLinalg.inv_ex(t.repr, false)(0))
    end inverse

    /**
     * Compute the Cholesky decomposition of a symmetric positive-definite matrix.
     * 
     * The Cholesky decomposition factors a symmetric positive-definite matrix A
     * into the product L * L^T, where L is a lower triangular matrix.
     * 
     * This implementation preserves differentiability by using tensor operations.
     * 
     * Args:
     *     t: Symmetric positive-definite matrix to decompose
     *     
     * Returns:
     *     Lower triangular matrix L such that t = L * L^T
     * 
     * Throws:
     *     Exception if the matrix is not positive-definite
     */
    def cholesky[A <: Singleton, B <: Singleton](
        t: Tensor2[A, B, Float32]
    ): Tensor2[A, B, Float32] =
        require(t.shape.dim1 == t.shape.dim2, "Matrix must be square for Cholesky decomposition")
        
        // val n = t.shape.dim1
        // val L = Tensor2(new Shape2[A, B](n, n), 0.0f)
        
        // // Use a block-wise approach that preserves the computation graph
        // for (i <- 0 until n) {
        //     val diagSum = (0 until i).foldLeft(Tensor0(0f) : Tensor0[Float32])(
        //         (acc, k) => 
        //             acc.add(L.apply(i, k).mul(L.apply(i, k)))                    
        //     )
        //     val diag_val = t.apply(i, i).sub(diagSum)
                
        //     if (diag_val.item <= 0) {
        //         throw new Exception("Matrix is not positive-definite")
        //     }
            
        //     // Update the diagonal element using sqrt which is differentiable
        //     L.update(i, i, diag_val.sqrt)
            
        //     // Update the column below the diagonal
        //     for (j <- i+1 until n) {
        //         val sum = (0 until i).foldLeft(Tensor0(0f) : Tensor0[Float32])((acc, k) => 
        //             acc.add(L.apply(j, k).mul(L.apply(i, k)))
        //         )
                
        //         L.update(j, i, (t.apply(j, i).sub(sum).div(L.apply(i, i))))
        //     }
        // }
        // L
        val (l,lt) = torch.linalg.BasicLinalg.cholesky_ex(t.repr, false, false) // Use the built-in cholesky function
        new Tensor2[A, B, Float32](new Shape2[A, B](l.shape(0), l.shape(1)), l)
    end cholesky



    // def det[A <: Singleton, B <: Singleton](t: Tensor2[A, B, Float32]) : Tensor0[Float32] =
    //     // implement determinant using gaussian elimination
    //     require(t.shape.dim1 == t.shape.dim2, "Matrix must be square to compute determinant")
    //     val n = t.shape.dim1
    //     var A = new Tensor2(t.shape, t.repr.detach()) // Create a copy of the matrix
    //     var determinantMultiplier = 1.0f

    //     for (col <- 0 until n) {
    //         // Find pivot row
    //         var pivotRow = col
    //         for (row <- col + 1 until n) {
    //         if (A(row, col).abs.item > A(pivotRow, col).abs.item) {
    //             pivotRow = row
    //         }
    //         }

    //         // Swap pivot row if necessary
    //         if (pivotRow != col) {
    //         A = A.swapRows(col, pivotRow)
    //         determinantMultiplier *= -1
    //         }

    //         // If the pivot element is zero, the matrix is singular
    //         if (A(col, col).abs.item < 1e-9f) {
    //         return Tensor0(0.0f)
    //         }

    //         // Eliminate entries below the pivot
    //         for (row <- col + 1 until n) {
    //         val factor = A(row, col).div(A(col, col))
    //         for (k <- col until n) {
    //             A = A.update(row, k, A(row, k).sub(factor.mul(A(col, k))))
    //         }
    //         }
    //     }

    //     // The determinant is the product of the diagonal entries multiplied by the determinantMultiplier
    //     val determinantU = (0 until n).foldLeft(Tensor0(1.0f): Tensor0[Float32])((acc, i) =>
    //         acc.mul(A(i, i))
    //     )
    //     determinantU.mul(Tensor0(determinantMultiplier))
    // end det
