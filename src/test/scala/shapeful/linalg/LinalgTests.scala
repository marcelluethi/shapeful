package shapeful.linalg

import scala.language.experimental.namedTypeArguments
import munit.FunSuite
import shapeful.*
import shapeful.tensor.DType
import shapeful.tensor.Shape.*
import shapeful.jax.Jax

class LinalgTests extends FunSuite:

  // Test fixtures - define common labels for reuse
  type Dim1 = "dim1"
  type Dim2 = "dim2"
  
  override def beforeAll(): Unit =
    // Initialize Python/JAX environment if needed
    super.beforeAll()

  test("inverse of identity matrix returns identity matrix") {
    // Create a 2x2 identity matrix
    val shape = Shape2[Dim1, Dim2](2, 2)
    val values = Seq(1.0f, 0.0f, 0.0f, 1.0f)
    val identityMatrix = Tensor(shape, values, DType.Float32)
    
    // Compute the inverse
    val inverse = Linalg.inverse(identityMatrix)
    
    // The inverse of identity should be identity
    assertEquals(inverse.shape.dims, Seq(2, 2))
    
    // Check equality using approxEquals
    assert(inverse.approxEquals(identityMatrix, tolerance = 1e-5f),
      s"Inverse of identity matrix should be identity.\nExpected: ${identityMatrix.inspect}\nActual: ${inverse.inspect}")
  }
  
  test("inverse of 2x2 matrix") {
    // Create a 2x2 matrix [[4, 7], [2, 6]]
    val shape = Shape2[Dim1, Dim2](2, 2)
    val values = Seq(4.0f, 7.0f, 2.0f, 6.0f)
    val matrix = Tensor(shape, values, DType.Float32)
    
    // Compute the inverse
    val inverse = Linalg.inverse(matrix)
    
    // The determinant is 4*6 - 7*2 = 24 - 14 = 10
    // So the inverse should be [[0.6, -0.7], [-0.2, 0.4]]
    assertEquals(inverse.shape.dims, Seq(2, 2))
    
    // Create the expected inverse matrix
    val expectedValues = Seq(0.6f, -0.7f, -0.2f, 0.4f)    
    val expected = Tensor(Shape2[Dim2, Dim1](2, 2), expectedValues, DType.Float32)
    
    // Check equality using approxEquals
    assert(inverse.approxEquals(expected, tolerance = 1e-5f),
      s"Inverse matrix incorrect.\nExpected: ${expected.inspect}\nActual: ${inverse.inspect}")
  }
  
  test("matrix multiplication with inverse gives identity") {
    // Create a 2x2 matrix
    val shape = Shape2[Dim1, Dim2](2, 2)
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f)
    val matrix = Tensor(shape, values, DType.Float32)
    
    // Compute the inverse
    val inverse = Linalg.inverse(matrix)
    
    // Multiply the matrix by its inverse using JAX directly
    val jaxResult = Jax.jnp.matmul(matrix.jaxValue, inverse.jaxValue)
    val resultShape = Shape2[Dim1, Dim1](2, 2)
    val result = new Tensor2[Dim1, Dim1](resultShape, jaxResult, DType.Float32)
    
    // Create a 2x2 identity matrix to compare with
    val idValues = Seq(1.0f, 0.0f, 0.0f, 1.0f)
    val identityMatrix = Tensor(resultShape, idValues, DType.Float32)
    
    // Check equality using approxEquals
    assert(result.approxEquals(identityMatrix, tolerance = 1e-5f),
      s"Matrix times its inverse should be identity.\nExpected: ${identityMatrix.inspect}\nActual: ${result.inspect}")
  }
  
  test("cholesky of identity matrix returns identity matrix") {
    // Create a 2x2 identity matrix
    val shape = Shape2[Dim1, Dim2](2, 2)
    val values = Seq(1.0f, 0.0f, 0.0f, 1.0f)
    val identityMatrix = Tensor(shape, values, DType.Float32)
    
    // Compute the Cholesky decomposition
    val chol = Linalg.cholesky(identityMatrix)
    
    // The Cholesky of identity should be identity
    assertEquals(chol.shape.dims, Seq(2, 2))
    
    // Check equality using approxEquals
    assert(chol.approxEquals(identityMatrix, tolerance = 1e-5f),
      s"Cholesky of identity matrix should be identity.\nExpected: ${identityMatrix.inspect}\nActual: ${chol.inspect}")
  }
  
  test("cholesky of positive definite matrix") {
    // Create a 2x2 positive definite matrix [[2, -1], [-1, 2]]
    val shape = Shape2[Dim1, Dim2](2, 2)
    val values = Seq(2.0f, -1.0f, -1.0f, 2.0f)
    val matrix = Tensor(shape, values, DType.Float32)
    
    // Compute the Cholesky decomposition
    val chol = Linalg.cholesky(matrix)
    
    // The Cholesky should be approximately [[sqrt(2), 0], [-1/sqrt(2), sqrt(3/2)]]
    assertEquals(chol.shape.dims, Seq(2, 2))
    
    // Check values with approxEquals
    val sqrt2 = math.sqrt(2).toFloat
    val sqrt3div2 = math.sqrt(1.5).toFloat
    
    // Create the expected Cholesky factor
    val expectedValues = Seq(sqrt2, 0.0f, -1.0f/sqrt2, sqrt3div2)
    val expected = Tensor(shape, expectedValues, DType.Float32)
    
    assert(chol.approxEquals(expected, tolerance = 1e-5f), 
      s"Cholesky decomposition incorrect.\nExpected: ${expected.inspect}\nActual: ${chol.inspect}")
  }
  
  test("cholesky decomposition reconstruction") {
    // Create a 3x3 positive definite matrix
    val shape = Shape2[Dim1, Dim2](3, 3)
    val values = Seq(
      4.0f, 1.0f, 1.0f,
      1.0f, 3.0f, 2.0f,
      1.0f, 2.0f, 6.0f
    )
    val matrix = Tensor(shape, values, DType.Float32)
    
    // Compute the Cholesky decomposition
    val chol = Linalg.cholesky(matrix)
    
    // Verify L * L.T ≈ original matrix
    // Calculate L * L.T using JAX
    val cholTranspose = Jax.jnp.transpose(chol.jaxValue)
    val reconstructed = Jax.jnp.matmul(chol.jaxValue, cholTranspose)
    val resultShape = Shape2[Dim1, Dim2](3, 3)
    val result = new Tensor2[Dim1, Dim2](resultShape, reconstructed, DType.Float32)
    
    // Check that the reconstruction is close to the original using approxEquals
    assert(result.approxEquals(matrix, tolerance = 1e-4f),
      s"Cholesky reconstruction incorrect.\nExpected: ${matrix.inspect}\nActual: ${result.inspect}")
  }
