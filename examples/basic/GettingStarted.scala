package examples.basic

import scala.language.experimental.namedTypeArguments
import shapeful.*

/**
 * Getting started with Shapeful tensors.
 * 
 * This example demonstrates:
 * - Creating tensors with different shapes
 * - Working with labeled dimensions  
 * - Basic tensor inspection
 */
object GettingStarted extends App:

  println("=== Shapeful Tensor Library - Getting Started ===\n")

  // Define some semantic labels for our tensor dimensions
  type Batch = "batch"
  type Feature = "feature" 
  type Height = "height"
  type Width = "width"

  // 1. Creating scalar tensors (0D)
  println("1. Scalar Tensors (0D)")
  val scalar = Tensor0(42.0f)
  println(s"Scalar tensor: $scalar")
  println(s"Value: ${scalar.toFloat}")
  println(s"Shape: ${scalar.shape.dims}")
  println()

  // 2. Creating vector tensors (1D)
  println("2. Vector Tensors (1D)")
  val vector = Tensor1[Feature](Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f))
  println(s"Vector tensor: $vector")
  println(s"Shape: ${vector.shape.dims}")
  println(s"Size: ${vector.shape.size}")
  println()

  // 3. Creating matrix tensors (2D)
  println("3. Matrix Tensors (2D)")
  val matrix = Tensor2[Batch, Feature](Seq(
    Seq(1.0f, 2.0f, 3.0f),
    Seq(4.0f, 5.0f, 6.0f),
    Seq(7.0f, 8.0f, 9.0f)
  ))
  println(s"Matrix tensor: $matrix")
  println(s"Shape: ${matrix.shape.dims} (${matrix.shape.dim[Batch]} batches, ${matrix.shape.dim[Feature]} features)")
  println()

  // 4. Creating 3D tensors 
  println("4. 3D Tensors")
  val tensor3d = Tensor3[Batch, Height, Width](Seq(
    Seq(
      Seq(1.0f, 2.0f),
      Seq(3.0f, 4.0f)
    ),
    Seq(
      Seq(5.0f, 6.0f), 
      Seq(7.0f, 8.0f)
    )
  ))
  println(s"3D tensor: $tensor3d")
  println(s"Shape: ${tensor3d.shape.dims} (${tensor3d.shape.dim[Batch]} × ${tensor3d.shape.dim[Height]} × ${tensor3d.shape.dim[Width]})")
  println()

  // 5. Special tensor creation methods
  println("5. Special Tensor Creation")
  
  val zeros = Tensor.zeros(Shape1[Feature](4))
  println(s"Zeros: $zeros")
  
  val ones = Tensor.ones(Shape1[Feature](3))
  println(s"Ones: $ones")
  
  println()

  // 6. Working with different data types
  println("6. Data Types")
  val intTensor = Tensor1[Feature](Seq(1.0f, 2.0f, 3.0f))
  println(s"Float32 tensor: $intTensor (${intTensor.dtype})")
  println("Note: DType conversion methods would be available in a complete implementation")
  println()

  // 7. Basic properties
  println("7. Tensor Properties")
  println(s"Matrix rank: ${matrix.shape.rank}")
  println(s"Matrix size: ${matrix.shape.size}")
  println(s"Matrix dims: ${matrix.shape.dims}")
  println()

  println("=== Getting Started Complete! ===")
