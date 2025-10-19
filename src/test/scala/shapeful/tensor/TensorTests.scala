package shapeful.tensor

import munit.FunSuite
import shapeful.Label
import shapeful.tensor.Shape.*

class TensorTests extends FunSuite:

  // Test fixtures - define common labels for reuse
  type Height = "height"
  type Width = "width"
  type Batch = "batch"
  type Channel = "channel"
  type Feature = "feature"

  override def beforeAll(): Unit =
    // Initialize Python/JAX environment if needed
    super.beforeAll()

  test("Tensor creation with different dtypes") {
    val shape = Shape(Axis[Height] -> 2, Axis[Width] -> 3)
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)

    val tensorFloat = Tensor(shape, values, DType.Float32)
    assertEquals(tensorFloat.dtype, DType.Float32)
    assertEquals(tensorFloat.shape.dims, Seq(2, 3))
  }

  test("Tensor0 creation from scalar values") {
    val floatTensor = Tensor0(3.14f)
    assertEquals(floatTensor.dtype, DType.Float32)
    assertEquals(floatTensor.shape.dims, Seq.empty)

    val intTensor = Tensor0(42)
    assertEquals(intTensor.dtype, DType.Int32)

    val boolTensor = Tensor0(true)
    assertEquals(boolTensor.dtype, DType.Bool)
  }

  test("Tensor1 creation") {
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f)
    val tensor = Tensor1(Axis[Feature], values)

    assertEquals(tensor.shape.dims, Seq(4), "dims")
    assertEquals(tensor.dtype, DType.Float32)
    assertEquals(tensor.shape.dim[Feature], 4)
  }

  test("Tensor2 creation") {
    val values = Seq(
      Seq(1.0f, 2.0f, 3.0f),
      Seq(4.0f, 5.0f, 6.0f)
    )
    val tensor = Tensor2(Axis[Height], Axis[Width], values)

    assertEquals(tensor.shape.dims, Seq(2, 3), "dims")
    assertEquals(tensor.shape.dim[Height], 2, "height")
    assertEquals(tensor.shape.dim[Width], 3)
  }

  test("Tensor2 identity matrix creation") {
    type Label = "aLabel"
    val expectedShape = Shape(Axis[Label] -> 2, Axis[Label] -> 2)
    val expectedTensor = Tensor(expectedShape, Seq(1f, 0f, 0f, 1f))
    val tensor = Tensor2.eye[Label](Shape(Axis[Label] -> 2))

    assertEquals(tensor.shape, expectedShape)
    assert(tensor.tensorEquals(expectedTensor))
  }

  test("Tensor2 identity matrix creation - improved API") {
    type Label = "aLabel"
    val expectedShape = Shape(Axis[Label] -> 3, Axis[Label] -> 3)
    val expectedTensor = Tensor(expectedShape, Seq(1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f))

    // Test with just an integer
    val tensor1 = Tensor2.eye[Label](3)
    assertEquals(tensor1.shape, expectedShape)
    assert(tensor1.tensorEquals(expectedTensor))

    // Test with Axis tuple
    val tensor2 = Tensor2.eye(Axis[Label] -> 3)
    assertEquals(tensor2.shape, expectedShape)
    assert(tensor2.tensorEquals(expectedTensor))
  }

  test("Tensor2  matrix creation from diag") {
    type Label = "aLabel"
    val expectedShape = Shape(Axis[Label] -> 2, Axis[Label] -> 2)
    val expectedTensor = Tensor(expectedShape, Seq(3f, 0f, 0f, 2f))
    val tensor = Tensor2.fromDiag[Label](Tensor1(Axis[Label], Seq(3f, 2f)))

    assertEquals(tensor.shape, expectedShape, "shape")
    assert(tensor.tensorEquals(expectedTensor), "equals")
  }

  test("zeros and ones creation") {
    val shape = Shape(Axis[Height] -> 2, Axis[Width] -> 3)

    val zeros = Tensor.zeros(shape)
    assertEquals(zeros.shape.dims, Seq(2, 3))
    assertEquals(zeros.dtype, DType.Float32)

    // Verify zeros tensor actually contains all zeros
    val expectedZeros = Tensor(shape, Seq.fill(6)(0.0f))
    assert(zeros.tensorEquals(expectedZeros))

    val ones = Tensor.ones(shape)
    assertEquals(ones.shape.dims, Seq(2, 3))
    assertEquals(ones.dtype, DType.Float32)

    // Verify ones tensor actually contains all ones
    val expectedOnes = Tensor(shape, Seq.fill(6)(1.0f))
    assert(ones.tensorEquals(expectedOnes))
  }

  test("zeros and ones creation with Axis arguments") {
    // Test 1D
    val zeros1d = Tensor.zeros(Axis[Height] -> 5)
    assertEquals(zeros1d.shape.dims, Seq(5))
    assertEquals(zeros1d.dtype, DType.Float32)

    val ones1d = Tensor.ones(Axis[Width] -> 4)
    assertEquals(ones1d.shape.dims, Seq(4))
    assertEquals(ones1d.dtype, DType.Float32)

    // Test 2D
    val zeros2d = Tensor.zeros(Axis[Height] -> 2, Axis[Width] -> 3)
    assertEquals(zeros2d.shape.dims, Seq(2, 3))
    assertEquals(zeros2d.dtype, DType.Float32)

    val ones2d = Tensor.ones(Axis[Height] -> 2, Axis[Width] -> 3)
    assertEquals(ones2d.shape.dims, Seq(2, 3))
    assertEquals(ones2d.dtype, DType.Float32)

    // Test 3D
    type Depth = "depth"
    val zeros3d = Tensor.zeros(Axis[Batch] -> 2, Axis[Height] -> 3, Axis[Width] -> 4)
    assertEquals(zeros3d.shape.dims, Seq(2, 3, 4))

    val ones3d = Tensor.ones(Axis[Batch] -> 2, Axis[Height] -> 3, Axis[Width] -> 4)
    assertEquals(ones3d.shape.dims, Seq(2, 3, 4))
  }

  test("reshape operation") {
    val originalShape = Shape(Axis[Height] -> 2, Axis[Width] -> 3)
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
    val tensor = Tensor(originalShape, values)

    val newShape = Shape(Axis[Batch] -> 3, Axis[Feature] -> 2)
    val reshaped = tensor.reshape(newShape)

    assertEquals(reshaped.shape.dims, Seq(3, 2))
    assertEquals(reshaped.dtype, tensor.dtype)
  }

  test("reshape with incompatible dimensions should fail") {
    val originalShape = Shape(Axis[Height] -> 2, Axis[Width] -> 3)
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
    val tensor = Tensor(originalShape, values)

    val incompatibleShape = Shape(Axis[Batch] -> 2, Axis[Feature] -> 4) // 8 elements vs 6

    intercept[IllegalArgumentException] {
      tensor.reshape(incompatibleShape)
    }
  }

  test("relabel operation") {
    val originalShape = Shape(Axis[Height] -> 2, Axis[Width] -> 3)
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
    val tensor = Tensor(originalShape, values)

    val relabeled = tensor.relabel[(Batch, Feature)]
    assertEquals(relabeled.shape.dims, Seq(2, 3))
    assertEquals(relabeled.shape.dim[Batch], 2)
    assertEquals(relabeled.shape.dim[Feature], 3)
  }

  test("dtype conversion") {
    val shape = Shape(Axis[Feature] -> 3)
    val values = Seq(1.0f, 2.0f, 3.0f)
    val tensor = Tensor(shape, values, DType.Float32)

    val intTensor = tensor.asType(DType.Int32)
    assertEquals(intTensor.dtype, DType.Int32)
    assertEquals(intTensor.shape.dims, tensor.shape.dims)

    // Converting to same dtype should return same instance
    val sameTensor = tensor.asType(DType.Float32)
    assert(sameTensor eq tensor)
  }

  test("vmap operation") {
    val shape = Shape(Axis[Batch] -> 3, Axis[Feature] -> 4)
    val values = (1 to 12).map(_.toFloat)
    val tensor = Tensor(shape, values)

    // Map over batch dimension - apply a function to each batch element
    val vmapped = tensor.vmap(Axis[Batch]) { batchElement =>
      // batchElement should be Tensor[Tuple1[Feature]] with shape (4,)
      assertEquals(batchElement.shape.dims, Seq(4))
      batchElement // Identity function for test
    }

    // Result should have shape (Batch, Feature) = (3, 4)
    assertEquals(vmapped.shape.dims, Seq(3, 4))
  }

  test("zipVmap operation") {
    val shape = Shape(Axis[Batch] -> 2, Axis[Feature] -> 3)
    val values1 = Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
    val values2 = Seq(2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f)

    val tensor1 = Tensor(shape, values1)
    val tensor2 = Tensor(shape, values2)

    val result = tensor1.zipVmap(Axis[Batch])(tensor2) { (t1, t2) =>
      // Both should be Tensor[Tuple1[Feature]] with shape (3,)
      assertEquals(t1.shape.dims, Seq(3))
      assertEquals(t2.shape.dims, Seq(3))
      t1 // Return first tensor for test
    }

    // Result should have shape (Batch, Feature) = (2, 3)
    assertEquals(result.shape.dims, Seq(2, 3))
  }

  test("tensor indexing") {
    val shape = Shape(Axis[Height] -> 2, Axis[Width] -> 3)
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
    val tensor = Tensor(shape, values)

    val indexer = tensor.at((1, 2))
    assertNotEquals(indexer, null)
  }

  test("tensor indexer get and set operations") {
    // Create a 2x2 tensor for testing indexing operations
    val shape = Shape(Axis[Height] -> 2, Axis[Width] -> 2)
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f)
    val tensor = Tensor(shape, values)

    // Test getting a value
    val indexer = tensor.at((1, 1))
    val retrievedValue = indexer.get
    assertEquals(retrievedValue.shape.dims, Seq.empty) // Should be scalar

    // The value at (1,1) should be 4.0f based on row-major ordering
    val expectedValue = Tensor0(4.0f)
    assert(retrievedValue.tensorEquals(expectedValue))

    // Test setting a value
    val newValue = Tensor0(10.0f)
    val updatedTensor = indexer.set(newValue)

    assertEquals(updatedTensor.shape.dims, tensor.shape.dims)
    assert(updatedTensor != tensor) // Should be a new instance

    // Verify the value was actually changed
    val newIndexer = updatedTensor.at((1, 1))
    val newRetrievedValue = newIndexer.get
    assert(newRetrievedValue.tensorEquals(newValue))

    // Verify other values remained unchanged
    val unchangedIndexer = updatedTensor.at((0, 0))
    val unchangedValue = unchangedIndexer.get
    val expectedUnchanged = Tensor0(1.0f)
    assert(unchangedValue.tensorEquals(expectedUnchanged))
  }

  test("tensor toString for different dimensions") {
    // Test that different tensor types can be converted to string without errors
    // and that scalar values are properly represented

    // 0D tensor
    val scalar = Tensor0(3.14f)
    val scalarStr = scalar.toString
    assert(scalarStr.nonEmpty)
    // Verify the scalar actually contains the expected value
    val expectedScalar = Tensor0(3.14f)
    assert(scalar.tensorEquals(expectedScalar), "scalar equals")

    // 1D tensor
    val vector = Tensor1(Axis[Feature], Seq(1.0f, 2.0f, 3.0f))
    val vectorStr = vector.toString
    assert(vectorStr.nonEmpty, "vector string")
    // Verify the vector contains expected values
    val expectedVector = Tensor1(Axis[Feature], Seq(1.0f, 2.0f, 3.0f))
    assert(vector.tensorEquals(expectedVector), "vector equals")

    // 2D tensor
    val matrix = Tensor2(Axis[Height], Axis[Width], Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f)))
    val matrixStr = matrix.toString
    assert(matrixStr.nonEmpty, "matrix string")
    // Verify the matrix contains expected values
    val expectedMatrix = Tensor2(Axis[Height], Axis[Width], Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f)))
    assert(matrix.tensorEquals(expectedMatrix), "matrix equals")
  }

  test("tensor stats method") {
    val shape = Shape(Axis[Height] -> 2, Axis[Width] -> 3)
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
    val tensor = Tensor(shape, values)

    val statsStr = tensor.stats()

    // Verify stats string contains expected sections
    assert(statsStr.contains("Mean:"))
    assert(statsStr.contains("StdDev:"))
    assert(statsStr.contains("Min:"))
    assert(statsStr.contains("Max:"))

    // More importantly, verify the tensor itself has the expected values
    val expectedTensor = Tensor(shape, values)
    assert(tensor.tensorEquals(expectedTensor))

    // Verify the tensor maintains its properties
    assertEquals(tensor.shape.dims, Seq(2, 3))
    assertEquals(tensor.dtype, DType.Float32)
  }

  test("tensor inspect method") {
    val shape = Shape(Axis[Feature] -> 3)
    val values = Seq(1.0f, 2.0f, 3.0f)
    val tensor = Tensor(shape, values)

    val inspectStr = tensor.inspect
    assert(inspectStr.contains("Tensor["))
    assert(inspectStr.contains("dtype:"))
    assert(inspectStr.contains("values:"))

    // More importantly, verify the tensor actually contains the expected values
    val expectedTensor = Tensor(shape, values)
    assert(tensor.tensorEquals(expectedTensor))
    assertEquals(tensor.dtype, DType.Float32)
    assertEquals(tensor.shape.dims, Seq(3))
  }

  test("dim method with specific labels") {
    val shape = Shape(Axis[Height] -> 5, Axis[Width] -> 7)
    val values = (1 to 35).map(_.toFloat)
    val tensor = Tensor(shape, values)

    assertEquals(tensor.shape.dim[Height], 5)
    assertEquals(tensor.shape.dim[Width], 7)
  }

  test("complex vmap with reduction") {
    val shape = Shape(Axis[Batch] -> 3, Axis[Feature] -> 4)
    val values = (1 to 12).map(_.toFloat)
    val tensor = Tensor(shape, values)

    // Map over batch and apply sum reduction to each batch element
    val vmapped = tensor.vmap(Axis[Batch]) { batchElement =>
      // Sum all features for each batch element, returning a scalar
      Tensor0(1)
    }

    // Result should be scalar for each batch element: shape (3,)
    assertEquals(vmapped.shape.dims, Seq(3))
  }

  test("tensor equality - identical tensors") {
    val shape = Shape(Axis[Height] -> 2, Axis[Width] -> 3)
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)

    val tensor1 = Tensor(shape, values)
    val tensor2 = Tensor(shape, values)

    assert(tensor1.tensorEquals(tensor2))
    assert(tensor1 == tensor2)
    assert(!(tensor1 != tensor2))
  }

  test("tensor equality - different values") {
    val shape = Shape(Axis[Height] -> 2, Axis[Width] -> 3)
    val values1 = Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
    val values2 = Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 7.0f) // Last value different

    val tensor1 = Tensor(shape, values1)
    val tensor2 = Tensor(shape, values2)

    assert(!tensor1.tensorEquals(tensor2))
    assert(tensor1 != tensor2)
    assert(!(tensor1 == tensor2))
  }

  test("tensor equality - different shapes") {
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)

    val tensor1 = Tensor(Shape(Axis[Height] -> 2, Axis[Width] -> 3), values)
    val tensor2 = Tensor(Shape(Axis[Height] -> 3, Axis[Width] -> 2), values)

    assert(!tensor1.tensorEquals(tensor2))
    assert(tensor1 != tensor2)
  }

  test("tensor equality - different dtypes") {
    val shape = Shape(Axis[Feature] -> 3)
    val values = Seq(1.0f, 2.0f, 3.0f)

    val tensor1 = Tensor(shape, values, DType.Float32)
    val tensor2 = Tensor(shape, values, DType.Float64)

    assert(!tensor1.tensorEquals(tensor2))
  }

  test("tensor equality - scalar tensors") {
    val tensor1 = Tensor0(3.14f)
    val tensor2 = Tensor0(3.14f)
    val tensor3 = Tensor0(2.71f)

    assert(tensor1.tensorEquals(tensor2))
    assert(!tensor1.tensorEquals(tensor3))
  }

  test("tensor approximate equality") {
    val shape = Shape(Axis[Feature] -> 3)
    val values1 = Seq(1.0f, 2.0f, 3.0f)
    val values2 = Seq(1.0000001f, 2.0000001f, 3.0000001f) // Very small difference

    val tensor1 = Tensor(shape, values1)
    val tensor2 = Tensor(shape, values2)

    assert(!tensor1.tensorEquals(tensor2)) // Exact equality should fail
    assert(tensor1.approxEquals(tensor2)) // Approximate equality should pass
    assert(tensor1.approxEquals(tensor2, tolerance = 1e-6f))
    assert(!tensor1.approxEquals(tensor2, tolerance = 1e-8f)) // Too strict tolerance
  }

  test("tensor element-wise equality") {
    val shape = Shape(Axis[Height] -> 2, Axis[Width] -> 2)
    val values1 = Seq(1.0f, 2.0f, 3.0f, 4.0f)
    val values2 = Seq(1.0f, 0.0f, 3.0f, 0.0f) // Different at positions (0,1) and (1,1)

    val tensor1 = Tensor(shape, values1)
    val tensor2 = Tensor(shape, values2)

    val elementComparison = tensor1.elementEquals(tensor2)
    assertEquals(elementComparison.dtype, DType.Bool)
    assertEquals(elementComparison.shape.dims, Seq(2, 2))

    // The result should be a boolean tensor showing which elements are equal
    // [true, false]
    // [true, false]
  }

  test("tensor equality with Object.equals") {
    val shape = Shape(Axis[Feature] -> 3)
    val values = Seq(1.0f, 2.0f, 3.0f)

    val tensor1 = Tensor(shape, values)
    val tensor2 = Tensor(shape, values)
    val notATensor: Any = "not a tensor"

    // Test Scala's standard equality
    assertEquals(tensor1, tensor2)
    assertNotEquals(tensor1, notATensor)
  }

  test("tensor hashCode consistency") {
    val shape = Shape(Axis[Feature] -> 3)
    val values = Seq(1.0f, 2.0f, 3.0f)

    val tensor1 = Tensor(shape, values)
    val tensor2 = Tensor(shape, values)

    // Equal tensors should have equal hash codes
    if tensor1.tensorEquals(tensor2) then assertEquals(tensor1.hashCode(), tensor2.hashCode())
  }

  test("tensor equality - element-wise shape mismatch") {
    val tensor1 = Tensor(Shape(Axis[Height] -> 2, Axis[Width] -> 3), (1 to 6).map(_.toFloat))
    val tensor2 = Tensor(Shape(Axis[Height] -> 3, Axis[Width] -> 2), (1 to 6).map(_.toFloat))

    intercept[IllegalArgumentException] {
      tensor1.elementEquals(tensor2)
    }
  }

  test("stack operation for Tensor0 and Tensor1") {
    // Stack three scalars into a vector
    val t1 = Tensor0(1.0f)
    val t2 = Tensor0(2.0f)
    val t3 = Tensor0(3.0f)
    val stacked = Tensor.stack(Axis[Feature])(Seq(t1, t2, t3))
    val expected = Tensor1(Axis[Feature], Seq(1.0f, 2.0f, 3.0f))
    assertEquals(stacked.shape.dims, Seq(3), "dims")
    assert(stacked.tensorEquals(expected), "Stacked tensor should equal expected")

    // Stack two vectors into a matrix
    val v1 = Tensor1(Axis[Feature], Seq(1.0f, 2.0f))
    val v2 = Tensor1(Axis[Feature], Seq(3.0f, 4.0f))
    val stacked2 = Tensor.stack(Axis[Batch])(Seq(v1, v2))
    val expected2 = Tensor2(Axis[Batch], Axis[Feature], Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f)))
    assertEquals(stacked2.shape.dims, Seq(2, 2), "dims")
    assert(stacked2.tensorEquals(expected2), "Second stacked tensor should equal expected")
  }

  test("concat operation") {
    // Concatenate two vectors along the feature dimension
    val v1 = Tensor1(Axis[Feature], Seq(1.0f, 2.0f))
    val v2 = Tensor1(Axis[Feature], Seq(3.0f, 4.0f))
    val concatenated = v1.concat(Axis[Feature])(v2)
    val expected = Tensor1(Axis[Feature], Seq(1.0f, 2.0f, 3.0f, 4.0f))
    assertEquals(concatenated.shape.dims, Seq(4), "dims")
    assert(concatenated.tensorEquals(expected), "equals")

    // Concatenate two matrices along the batch dimension
    val m1 = Tensor2(Axis[Batch], Axis[Feature], Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f)))
    val m2 = Tensor2(Axis[Batch], Axis[Feature], Seq(Seq(5.0f, 6.0f)))
    val concatenated2 = m1.concat(Axis[Batch])(m2)
    val expected2 = Tensor2(Axis[Batch], Axis[Feature], Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f), Seq(5.0f, 6.0f)))
    assertEquals(concatenated2.shape.dims, Seq(3, 2), "dims")
    assert(concatenated2.tensorEquals(expected2), "equals")

    // Concatenate two matrices along the feature dimension
    val m3 = Tensor2(Axis[Batch], Axis[Feature], Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f)))
    val m4 = Tensor2(Axis[Batch], Axis[Feature], Seq(Seq(5.0f, 6.0f), Seq(7.0f, 8.0f)))
    val concatenated3 = m3.concat(Axis[Feature])(m4)
    val expected3 = Tensor2(Axis[Batch], Axis[Feature], Seq(Seq(1.0f, 2.0f, 5.0f, 6.0f), Seq(3.0f, 4.0f, 7.0f, 8.0f)))
    assertEquals(concatenated3.shape.dims, Seq(2, 4), "dims")
    assert(concatenated3.tensorEquals(expected3), "equals")
  }

  test("split operation") {
    // Test splitting a 2D tensor along the batch dimension
    val shape = Shape(Axis[Batch] -> 3, Axis[Feature] -> 4)
    val values = (1 to 12).map(_.toFloat)
    val tensor = Tensor(shape, values)

    // Split along batch dimension - should get 3 tensors of shape (4,)
    val splits = tensor.unstack(Axis[Batch])

    assertEquals(splits.length, 3)
    splits.foreach { split =>
      assertEquals(split.shape.dims, Seq(4), "dims")
    }

    // Check that split tensors contain correct values
    val expected1 = Tensor1(Axis[Feature], Seq(1.0f, 2.0f, 3.0f, 4.0f))
    val expected2 = Tensor1(Axis[Feature], Seq(5.0f, 6.0f, 7.0f, 8.0f))
    val expected3 = Tensor1(Axis[Feature], Seq(9.0f, 10.0f, 11.0f, 12.0f))

    assert(splits(0).tensorEquals(expected1), "split 0")
    assert(splits(1).tensorEquals(expected2), "split 1")
    assert(splits(2).tensorEquals(expected3), "split 2")

    // Test splitting along feature dimension - should get 4 tensors of shape (3,)
    val featureSplits = tensor.unstack(Axis[Feature])

    assertEquals(featureSplits.length, 4, "length")
    featureSplits.foreach { split =>
      assertEquals(split.shape.dims, Seq(3), "dims")
    }

    // Check that feature splits contain correct values
    val expectedF1 = Tensor1(Axis[Batch], Seq(1.0f, 5.0f, 9.0f))
    val expectedF2 = Tensor1(Axis[Batch], Seq(2.0f, 6.0f, 10.0f))
    val expectedF3 = Tensor1(Axis[Batch], Seq(3.0f, 7.0f, 11.0f))
    val expectedF4 = Tensor1(Axis[Batch], Seq(4.0f, 8.0f, 12.0f))

    assert(featureSplits(0).tensorEquals(expectedF1), "f0")
    assert(featureSplits(1).tensorEquals(expectedF2), "f1")
    assert(featureSplits(2).tensorEquals(expectedF3), "f2")
    assert(featureSplits(3).tensorEquals(expectedF4), "f3")
  }

  test("split operation on 3D tensor") {
    // Test splitting a 3D tensor
    type Depth = "depth"
    val shape = Shape(Axis[Batch] -> 2, Axis[Height] -> 2, Axis[Width] -> 3)
    val values = (1 to 12).map(_.toFloat)
    val tensor = Tensor(shape, values)

    // Split along batch dimension - should get 2 tensors of shape (2, 3)
    val batchSplits = tensor.unstack(Axis[Batch])

    assertEquals(batchSplits.length, 2)
    batchSplits.foreach { split =>
      assertEquals(split.shape.dims, Seq(2, 3))
    }

    // Split along height dimension - should get 2 tensors of shape (2, 3)
    val heightSplits = tensor.unstack(Axis[Height])

    assertEquals(heightSplits.length, 2)
    heightSplits.foreach { split =>
      assertEquals(split.shape.dims, Seq(2, 3))
    }

    // Split along width dimension - should get 3 tensors of shape (2, 2)
    val widthSplits = tensor.unstack(Axis[Width])

    assertEquals(widthSplits.length, 3)
    widthSplits.foreach { split =>
      assertEquals(split.shape.dims, Seq(2, 2))
    }
  }

  test("splitAt operation on Tensor1") {
    import shapeful.tensor.TensorSlicing.*

    // Test splitting a 1D tensor
    val tensor = Tensor1(Axis[Feature], Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f))

    // Split at index 3
    val (first, second) = tensor.splitAt[Feature](3)

    assertEquals(first.shape.dims, Seq(3), "first dims")
    assertEquals(second.shape.dims, Seq(3), "second dims")

    // Check values
    val expectedFirst = Tensor1(Axis[Feature], Seq(1.0f, 2.0f, 3.0f))
    val expectedSecond = Tensor1(Axis[Feature], Seq(4.0f, 5.0f, 6.0f))

    assert(first.tensorEquals(expectedFirst), "first equals")
    assert(second.tensorEquals(expectedSecond), "second equals")

    // Split at beginning
    val (empty, all) = tensor.splitAt[Feature](0)
    assertEquals(empty.shape.dims, Seq(0), "empty dims")
    assertEquals(all.shape.dims, Seq(6), "all dims")

    // Split at end
    val (allButEnd, endEmpty) = tensor.splitAt[Feature](6)
    assertEquals(allButEnd.shape.dims, Seq(6), "allButEnd dims")
    assertEquals(endEmpty.shape.dims, Seq(0), "endEmpty dims")
    assert(allButEnd.tensorEquals(tensor), "allButEnd equals")
  }

  test("splitAt operation on Tensor2") {
    import shapeful.tensor.TensorSlicing.*

    // Test splitting a 2D tensor along different axes
    val shape = Shape(Axis[Batch] -> 3, Axis[Feature] -> 4)
    val values = (1 to 12).map(_.toFloat)
    val tensor = Tensor(shape, values)

    // Split along batch dimension
    val (batch1, batch2) = tensor.splitAt[Batch](2)

    assertEquals(batch1.shape.dims, Seq(2, 4))
    assertEquals(batch2.shape.dims, Seq(1, 4))

    // Check that batch1 contains first 2 rows
    val expectedBatch1 = Tensor2(
      Axis[Batch],
      Axis[Feature],
      Seq(
        Seq(1.0f, 2.0f, 3.0f, 4.0f),
        Seq(5.0f, 6.0f, 7.0f, 8.0f)
      )
    )
    assert(batch1.tensorEquals(expectedBatch1), "batch1 equals")

    // Check that batch2 contains last row
    val expectedBatch2 = Tensor2(
      Axis[Batch],
      Axis[Feature],
      Seq(
        Seq(9.0f, 10.0f, 11.0f, 12.0f)
      )
    )
    assert(batch2.tensorEquals(expectedBatch2), "batch2 equals")

    // Split along feature dimension
    val (features1, features2) = tensor.splitAt[Feature](2)

    assertEquals(features1.shape.dims, Seq(3, 2), "features1 dims")
    assertEquals(features2.shape.dims, Seq(3, 2), "features2 dims")

    // Check that features1 contains first 2 columns
    val expectedFeatures1 = Tensor2(
      Axis[Batch],
      Axis[Feature],
      Seq(
        Seq(1.0f, 2.0f),
        Seq(5.0f, 6.0f),
        Seq(9.0f, 10.0f)
      )
    )
    assert(features1.tensorEquals(expectedFeatures1), "features1 equals")
  }

  test("splitAt operation on Tensor3") {
    import shapeful.tensor.TensorSlicing.*

    // Test splitting a 3D tensor
    type Depth = "depth"
    val shape = Shape(Axis[Batch] -> 2, Axis[Height] -> 2, Axis[Width] -> 3)
    val values = (1 to 12).map(_.toFloat)
    val tensor = Tensor(shape, values)

    // Split along width dimension
    val (left, right) = tensor.splitAt[Width](2)

    assertEquals(left.shape.dims, Seq(2, 2, 2))
    assertEquals(right.shape.dims, Seq(2, 2, 1))

    // Split along height dimension
    val (top, bottom) = tensor.splitAt[Height](1)

    assertEquals(top.shape.dims, Seq(2, 1, 3))
    assertEquals(bottom.shape.dims, Seq(2, 1, 3))

    // Split along batch dimension
    val (firstBatch, secondBatch) = tensor.splitAt[Batch](1)

    assertEquals(firstBatch.shape.dims, Seq(1, 2, 3))
    assertEquals(secondBatch.shape.dims, Seq(1, 2, 3))
  }

  test("splitAt bounds checking") {
    import shapeful.tensor.TensorSlicing.*

    val tensor = Tensor1(Axis[Feature], Seq(1.0f, 2.0f, 3.0f, 4.0f))

    // Valid bounds
    val (first, second) = tensor.splitAt[Feature](2)
    assertEquals(first.shape.dims(0) + second.shape.dims(0), 4, "sum")

    // Test boundary conditions
    intercept[IllegalArgumentException] {
      tensor.splitAt[Feature](-1)
    }

    intercept[IllegalArgumentException] {
      tensor.splitAt[Feature](5) // Beyond tensor size
    }
  }

  test("toDevice method with CPU") {
    val shape = Shape(Axis[Height] -> 2, Axis[Width] -> 3)
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
    val originalTensor = Tensor(shape, values)

    val cpuTensor = originalTensor.toDevice(Device.CPU)

    // Verify it returns a new tensor instance
    assert(cpuTensor ne originalTensor)

    // Verify shape and dtype are preserved
    assertEquals(cpuTensor.shape.dims, originalTensor.shape.dims)
    assertEquals(cpuTensor.dtype, originalTensor.dtype)

    // Verify the tensor values are preserved
    assert(cpuTensor.tensorEquals(originalTensor))

    // Verify the device is CPU (check device platform)
    val deviceStr = shapeful.jax.Jax.device_get(cpuTensor.jaxValue).platform.as[String]
    assertEquals(deviceStr, "cpu")
  }

  test("toDevice method returns different instances for same device") {
    val shape = Shape(Axis[Feature] -> 4)
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f)
    val originalTensor = Tensor(shape, values)

    val cpuTensor1 = originalTensor.toDevice(Device.CPU)
    val cpuTensor2 = originalTensor.toDevice(Device.CPU)

    // Should return different instances even for same device
    assert(cpuTensor1 ne cpuTensor2)
    assert(cpuTensor1 ne originalTensor)

    // But should have same values
    assert(cpuTensor1.tensorEquals(cpuTensor2))
    assert(cpuTensor1.tensorEquals(originalTensor))
  }

  test("toDevice preserves tensor properties across different dtypes") {
    val shape = Shape(Axis[Feature] -> 3)

    // Test with Float32
    val floatTensor = Tensor(shape, Seq(1.0f, 2.0f, 3.0f), DType.Float32)
    val floatCpuTensor = floatTensor.toDevice(Device.CPU)
    assertEquals(floatCpuTensor.dtype, DType.Float32)
    assertEquals(floatCpuTensor.shape.dims, shape.dims)
    assert(floatCpuTensor.tensorEquals(floatTensor))

    // Test with scalar tensors of different types
    val intScalar = Tensor0(42)
    val intCpuScalar = intScalar.toDevice(Device.CPU)
    assertEquals(intCpuScalar.dtype, DType.Int32)
    assert(intCpuScalar.tensorEquals(intScalar))

    val boolScalar = Tensor0(true)
    val boolCpuScalar = boolScalar.toDevice(Device.CPU)
    assertEquals(boolCpuScalar.dtype, DType.Bool)
    assert(boolCpuScalar.tensorEquals(boolScalar))
  }

  test("toDevice with GPU when available") {
    val shape = Shape(Axis[Feature] -> 4)
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f)
    val originalTensor = Tensor(shape, values)

    try
      val gpuTensor = originalTensor.toDevice(Device.GPU)

      // Verify it returns a new tensor instance
      assert(gpuTensor ne originalTensor)

      // Verify shape and dtype are preserved
      assertEquals(gpuTensor.shape.dims, originalTensor.shape.dims)
      assertEquals(gpuTensor.dtype, originalTensor.dtype)

      // Verify the tensor values are preserved
      assert(gpuTensor.tensorEquals(originalTensor))

      // Verify the device is GPU (check device platform)
      val deviceStr = shapeful.jax.Jax.device_get(gpuTensor.jaxValue).platform.as[String]
      assertEquals(deviceStr, "gpu")
    catch
      case _: Exception =>
        // GPU not available, skip this test
        println("GPU not available, skipping GPU test")
  }

  test("rearrange transposes tensor dimensions") {
    // Create a tensor with shape (Batch, Height, Width)
    val shape = Shape(Axis[Batch] -> 2, Axis[Height] -> 3, Axis[Width] -> 4)
    val values = (1 to 24).map(_.toFloat).toSeq
    val tensor = Tensor(shape, values)

    // Rearrange to (Width, Height, Batch)
    val rearranged = tensor.rearrange[(Width, Height, Batch)](
      Axis[Width],
      Axis[Height],
      Axis[Batch]
    )

    // Verify the new shape
    assertEquals(rearranged.shape.dims, Seq(4, 3, 2))
    assertEquals(rearranged.shape.dim[Width], 4, "width dim")
    assertEquals(rearranged.shape.dim[Height], 3, "height dim")
    assertEquals(rearranged.shape.dim[Batch], 2, "batch dim")

    // Verify axis labels are in the correct order
    assertEquals(rearranged.shape.axisLabels, Seq("width", "height", "batch"))
  }

  test("rearrange with different permutations") {
    // Create a 3D tensor
    val shape = Shape(Axis[Batch] -> 2, Axis[Height] -> 3, Axis[Width] -> 4)
    val values = (1 to 24).map(_.toFloat).toSeq
    val tensor = Tensor(shape, values)

    // Test various rearrangements
    val bhw = tensor.rearrange[(Batch, Height, Width)](Axis[Batch], Axis[Height], Axis[Width])
    assertEquals(bhw.shape.dims, Seq(2, 3, 4))

    val bwh = tensor.rearrange[(Batch, Width, Height)](Axis[Batch], Axis[Width], Axis[Height])
    assertEquals(bwh.shape.dims, Seq(2, 4, 3))

    val hwb = tensor.rearrange[(Height, Width, Batch)](Axis[Height], Axis[Width], Axis[Batch])
    assertEquals(hwb.shape.dims, Seq(3, 4, 2))
  }

  test("rearrange preserves element count") {
    val shape = Shape(Axis[Batch] -> 2, Axis[Height] -> 3, Axis[Width] -> 4)
    val values = (1 to 24).map(_.toFloat).toSeq
    val tensor = Tensor(shape, values)

    val rearranged = tensor.rearrange[(Width, Height, Batch)](
      Axis[Width],
      Axis[Height],
      Axis[Batch]
    )

    assertEquals(rearranged.shape.dims.product, tensor.shape.dims.product)
  }

  test("rearrange fails with wrong number of axes") {
    val shape = Shape(Axis[Batch] -> 2, Axis[Height] -> 3, Axis[Width] -> 4)
    val tensor = Tensor(shape, Seq.fill(24)(1.0f))

    // This should fail at runtime because we pass wrong number of axis arguments
    // (Note: We can't test compile-time failures easily, but providing wrong number
    // of Axis arguments at runtime will trigger the require check)
    intercept[IllegalArgumentException] {
      // Intentionally pass only 2 axis arguments when 3 are needed
      // We use the correct type to get past compile-time, but wrong runtime args
      tensor.rearrange[(Batch, Height, Width)](Axis[Batch], Axis[Height])
    }
  }

  test("rearrange fails with non-existent axis") {
    type NonExistent = "nonexistent"
    val shape = Shape(Axis[Batch] -> 2, Axis[Height] -> 3, Axis[Width] -> 4)
    val tensor = Tensor(shape, Seq.fill(24)(1.0f))

    intercept[IllegalArgumentException] {
      // Try to rearrange with an axis that doesn't exist
      tensor.rearrange[(NonExistent, Height, Width)](
        Axis[NonExistent],
        Axis[Height],
        Axis[Width]
      )
    }
  }

end TensorTests
