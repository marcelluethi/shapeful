package examples.basic

import scala.language.experimental.namedTypeArguments
import shapeful.*

/**
 * Basic tensor arithmetic operations.
 * 
 * This example demonstrates:
 * - Element-wise operations
 * - Broadcasting behavior
 * - Reduction operations
 * - Mathematical functions
 */
object ArithmeticOps extends App:

  println("=== Tensor Arithmetic Operations ===\n")

  // Define labels
  type Feature = "feature"
  type Batch = "batch"

  // 1. Element-wise operations
  println("1. Element-wise Operations")
  
  val a = Tensor1[Feature](Seq(1.0f, 2.0f, 3.0f, 4.0f))
  val b = Tensor1[Feature](Seq(2.0f, 3.0f, 4.0f, 5.0f))
  
  println(s"a = $a")
  println(s"b = $b")
  println(s"a + b = ${a + b}")
  println(s"a - b = ${a - b}")
  println(s"a * b = ${a * b}")
  println(s"a / b = ${a / b}")
  println()

  // 2. Scalar operations
  println("2. Scalar Operations")
  
  val scalar = Tensor0(2.0f)
  println(s"scalar = $scalar")
  println(s"a + scalar = ${a + scalar}")
  println(s"a * scalar = ${a * scalar}")
  println(s"a / scalar = ${a / scalar}")
  println()

  // 3. Matrix operations
  println("3. Matrix Operations")
  
  val matrix1 = Tensor2[Batch, Feature](Seq(
    Seq(1.0f, 2.0f, 3.0f),
    Seq(4.0f, 5.0f, 6.0f)
  ))
  
  val matrix2 = Tensor2[Batch, Feature](Seq(
    Seq(2.0f, 2.0f, 2.0f),
    Seq(3.0f, 3.0f, 3.0f)
  ))
  
  println(s"matrix1 = $matrix1")
  println(s"matrix2 = $matrix2")
  println(s"matrix1 + matrix2 = ${matrix1 + matrix2}")
  println(s"matrix1 * matrix2 = ${matrix1 * matrix2}")
  println()

  // 4. Mathematical functions
  println("4. Mathematical Functions")
  
  val values = Tensor1[Feature](Seq(0.0f, 1.0f, 2.0f, 3.0f))
  println(s"values = $values")
  println(s"exp(values) = ${values.exp}")
  println(s"sin(values) = ${values.sin}")
  println(s"cos(values) = ${values.cos}")
  println(s"tanh(values) = ${values.tanh}")
  println()

  // 5. Reduction operations
  println("5. Reduction Operations")
  
  val data = Tensor2[Batch, Feature](Seq(
    Seq(1.0f, 2.0f, 3.0f, 4.0f),
    Seq(5.0f, 6.0f, 7.0f, 8.0f),
    Seq(9.0f, 10.0f, 11.0f, 12.0f)
  ))
  
  println(s"data = $data")
  println(s"sum = ${data.sum}")
  println(s"mean = ${data.mean}")
  println(s"max = ${data.max}")
  println(s"min = ${data.min}")
  println()

  // 6. Comparisons and boolean operations
  println("6. Comparisons")
  
  val x = Tensor1[Feature](Seq(1.0f, 2.0f, 3.0f, 4.0f))
  val threshold = Tensor1[Feature](Seq(2.5f, 2.5f, 2.5f, 2.5f))  // Same shape for comparison
  
  println(s"x = $x")
  println(s"threshold = $threshold")
  println(s"x > threshold = ${x > threshold}")
  println(s"x == x = ${x == x}")
  println()

  // 7. Chaining operations
  println("7. Chaining Operations")
  
  val input = Tensor1[Feature](Seq(-2.0f, -1.0f, 0.0f, 1.0f, 2.0f))
  val result = input
    .exp           // Exponential
    .tanh          // Hyperbolic tangent
    * Tensor0(10.0f) // Scale by 10
    + Tensor0(1.0f)  // Add bias
  
  println(s"input = $input")
  println(s"exp(input).tanh() * 10 + 1 = $result")
  println()

  println("=== Arithmetic Operations Complete! ===")
