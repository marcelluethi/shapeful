package examples.basic

import shapeful.*

/** Getting started with Shapeful tensors.
  *
  * This example demonstrates:
  *   - Creating tensors with labeled dimensions
  *   - Inspecting shapes with meaningful axis labels
  *   - Basic tensor operations
  */
object GettingStarted extends App:

  println("=== Shapeful Tensor Library - Getting Started ===\n")

  // Define semantic labels for our tensor dimensions
  // Using type aliases enables IDE refactoring support
  type Batch = "Batch"
  type Feature = "Feature"
  type Height = "Height"
  type Width = "Width"
  type Channel = "Channel"
  type Row = "Row"

  // 1. Creating scalar tensors (0D)
  println("1. Scalar Tensors (0D)")
  val scalar = Tensor0(42.0f)
  println(s"Scalar tensor: $scalar")
  println(s"Value: ${scalar.toFloat}")
  println(s"Shape: ${scalar.shape}")
  println()

  // 2. Creating vector tensors (1D) with labeled axis
  println("2. Vector Tensors (1D)")
  val vector = Tensor1(Axis[Feature], Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f))
  println(s"Vector tensor: $vector")
  println(s"Shape: ${vector.shape}")  // Shows Shape(Feature=5)
  println(s"Size: ${vector.shape.size}")
  println()

  // 3. Creating matrix tensors (2D) with labeled axes
  println("3. Matrix Tensors (2D)")
  val matrix = Tensor2(Axis[Batch], Axis[Feature],
    Seq(
      Seq(1.0f, 2.0f, 3.0f),
      Seq(4.0f, 5.0f, 6.0f),
      Seq(7.0f, 8.0f, 9.0f)
    )
  )
  println(s"Matrix tensor: $matrix")
  println(s"Shape: ${matrix.shape}")  // Shows Shape(Batch=3, Feature=3)
  println(s"Batch dimension: ${matrix.shape.dim(Axis[Batch])}")
  println(s"Feature dimension: ${matrix.shape.dim(Axis[Feature])}")
  println()

  // 4. Creating 3D tensors with labeled axes
  println("4. 3D Tensors")
  val tensor3d = Tensor3(Axis[Batch], Axis[Height], Axis[Width],
    Seq(
      Seq(
        Seq(1.0f, 2.0f),
        Seq(3.0f, 4.0f)
      ),
      Seq(
        Seq(5.0f, 6.0f),
        Seq(7.0f, 8.0f)
      )
    )
  )
  println(s"3D tensor: $tensor3d")
  println(s"Shape: ${tensor3d.shape}")  // Shows Shape(Batch=2, Height=2, Width=2)
  println()

  // 5. Special tensor creation methods with Axis labels
  println("5. Special Tensor Creation with Shape API")

  val zeros = Tensor.zeros(Shape(Axis[Feature] -> 4))
  println(s"Zeros: $zeros")
  println(s"Shape: ${zeros.shape}")  // Shows Shape(Feature=4)

  val ones = Tensor.ones(Shape(Axis[Channel] -> 3))
  val onesOther = Tensor.ones(Axis[Channel] -> 3)  // Alternative syntax
  println(s"Ones: $ones")
  println(s"Shape: ${ones.shape}")  // Shows Shape(Channel=3)

  val identity = Tensor2.eye(Shape1(Axis[Row] -> 3))
  println(s"Identity matrix: $identity")
  println(s"Shape: ${identity.shape}")  // Shows Shape(Row=3, Row=3)
  println()

  // 6. Working with different data types
  println("6. Data Types")
  val floatTensor = Tensor1(Axis[Feature], Seq(1.0f, 2.0f, 3.0f))
  println(s"Float32 tensor: $floatTensor")
  println(s"DType: ${floatTensor.dtype}")
  println(s"Shape: ${floatTensor.shape}")  // Shows Shape(Feature=3)

  // Convert to different dtype
  val doubleTensor = floatTensor.asType(DType.Float64)
  println(s"Float64 tensor: $doubleTensor (${doubleTensor.dtype})")

  val intTensor = floatTensor.asType(DType.Int32)
  println(s"Int32 tensor: $intTensor (${intTensor.dtype})")
  println()

  // 7. Basic properties and inspection
  println("7. Tensor Properties")
  println(s"Matrix rank: ${matrix.shape.rank}")
  println(s"Matrix size: ${matrix.shape.size}")
  println(s"Matrix dimensions: ${matrix.shape.dims}")
  println(s"Matrix axis labels: ${matrix.shape.axisLabels}")
  println()

  // 8. Working with axis labels
  println("8. Axis Label Examples")
  val image = Tensor3(Axis[Batch], Axis[Height], Axis[Width],
    Seq.fill(1)(Seq.fill(28)(Seq.fill(28)(0.5f)))
  )
  println(s"Image tensor shape: ${image.shape}")  // Shows Shape(Batch=1, Height=28, Width=28)
  println(s"Height: ${image.shape.dim(Axis[Height])}")
  println(s"Width: ${image.shape.dim(Axis[Width])}")
  println()

  println("=== Getting Started Complete! ===")
  println("\nKey Takeaway: All shapes now display with meaningful labels!")
  println("Example: Shape(Batch=3, Feature=3)")
  println("This makes your code self-documenting and easier to understand.")
  println("\nTip: Using type aliases (type Batch = \"Batch\") provides IDE refactoring support!")
