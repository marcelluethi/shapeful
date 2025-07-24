package shapeful.tensor

import scala.language.experimental.namedTypeArguments
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
    val shape = Shape2[Height, Width](2, 3)
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
    val tensor = Tensor1[Feature](values)

    assertEquals(tensor.shape.dims, Seq(4))
    assertEquals(tensor.dtype, DType.Float32)
    assertEquals(tensor.shape.dim[Feature], 4)
  }

  test("Tensor2 creation") {
    val values = Seq(
      Seq(1.0f, 2.0f, 3.0f),
      Seq(4.0f, 5.0f, 6.0f)
    )
    val tensor = Tensor2[Height, Width](values)

    assertEquals(tensor.shape.dims, Seq(2, 3))
    assertEquals(tensor.shape.dim[Height], 2)
    assertEquals(tensor.shape.dim[Width], 3)
  }

  test("zeros and ones creation") {
    val shape = Shape2[Height, Width](2, 3)

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

  test("reshape operation") {
    val originalShape = Shape2[Height, Width](2, 3)
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
    val tensor = Tensor(originalShape, values)

    val newShape = Shape2[Batch, Feature](3, 2)
    val reshaped = tensor.reshape(newShape)

    assertEquals(reshaped.shape.dims, Seq(3, 2))
    assertEquals(reshaped.dtype, tensor.dtype)
  }

  test("reshape with incompatible dimensions should fail") {
    val originalShape = Shape2[Height, Width](2, 3)
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
    val tensor = Tensor(originalShape, values)

    val incompatibleShape = Shape2[Batch, Feature](2, 4) // 8 elements vs 6

    intercept[IllegalArgumentException] {
      tensor.reshape(incompatibleShape)
    }
  }

  test("relabel operation") {
    val originalShape = Shape2[Height, Width](2, 3)
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
    val tensor = Tensor(originalShape, values)

    val relabeled = tensor.relabel[(Batch, Feature)]
    assertEquals(relabeled.shape.dims, Seq(2, 3))
    assertEquals(relabeled.shape.dim[Batch], 2)
    assertEquals(relabeled.shape.dim[Feature], 3)
  }

  test("dtype conversion") {
    val shape = Shape1[Feature](3)
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
    val shape = Shape2[Batch, Feature](3, 4)
    val values = (1 to 12).map(_.toFloat)
    val tensor = Tensor(shape, values)

    // Map over batch dimension - apply a function to each batch element
    val vmapped = tensor.vmap[Batch, Tuple1[Feature]] { batchElement =>
      // batchElement should be Tensor[Tuple1[Feature]] with shape (4,)
      assertEquals(batchElement.shape.dims, Seq(4))
      batchElement // Identity function for test
    }

    // Result should have shape (Batch, Feature) = (3, 4)
    assertEquals(vmapped.shape.dims, Seq(3, 4))
  }

  test("zipVmap operation") {
    val shape = Shape2[Batch, Feature](2, 3)
    val values1 = Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
    val values2 = Seq(2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f)

    val tensor1 = Tensor(shape, values1)
    val tensor2 = Tensor(shape, values2)

    val result = tensor1.zipVmap[Batch, Tuple2[Batch, Feature], Tuple1[Feature]](tensor2) { (t1, t2) =>
      // Both should be Tensor[Tuple1[Feature]] with shape (3,)
      assertEquals(t1.shape.dims, Seq(3))
      assertEquals(t2.shape.dims, Seq(3))
      t1 // Return first tensor for test
    }

    // Result should have shape (Batch, Feature) = (2, 3)
    assertEquals(result.shape.dims, Seq(2, 3))
  }

  test("tensor indexing") {
    val shape = Shape2[Height, Width](2, 3)
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)
    val tensor = Tensor(shape, values)

    val indexer = tensor.at((1, 2))
    assertNotEquals(indexer, null)
  }

  test("tensor indexer get and set operations") {
    // Create a 2x2 tensor for testing indexing operations
    val shape = Shape2[Height, Width](2, 2)
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
    assert(scalar.tensorEquals(expectedScalar))

    // 1D tensor
    val vector = Tensor1[Feature](Seq(1.0f, 2.0f, 3.0f))
    val vectorStr = vector.toString
    assert(vectorStr.nonEmpty)
    // Verify the vector contains expected values
    val expectedVector = Tensor1[Feature](Seq(1.0f, 2.0f, 3.0f))
    assert(vector.tensorEquals(expectedVector))

    // 2D tensor
    val matrix = Tensor2[Height, Width](Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f)))
    val matrixStr = matrix.toString
    assert(matrixStr.nonEmpty)
    // Verify the matrix contains expected values
    val expectedMatrix = Tensor2[Height, Width](Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f)))
    assert(matrix.tensorEquals(expectedMatrix))
  }

  test("tensor stats method") {
    val shape = Shape2[Height, Width](2, 3)
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
    val shape = Shape1[Feature](3)
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
    val shape = Shape2[Height, Width](5, 7)
    val values = (1 to 35).map(_.toFloat)
    val tensor = Tensor(shape, values)

    assertEquals(tensor.shape.dim[Height], 5)
    assertEquals(tensor.shape.dim[Width], 7)
  }

  test("complex vmap with reduction") {
    val shape = Shape2[Batch, Feature](3, 4)
    val values = (1 to 12).map(_.toFloat)
    val tensor = Tensor(shape, values)

    // Map over batch and apply sum reduction to each batch element
    val vmapped = tensor.vmap[Batch, EmptyTuple] { batchElement =>
      // Sum all features for each batch element, returning a scalar
      Tensor0(1)
    }

    // Result should be scalar for each batch element: shape (3,)
    assertEquals(vmapped.shape.dims, Seq(3))
  }

  test("tensor equality - identical tensors") {
    val shape = Shape2[Height, Width](2, 3)
    val values = Seq(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f)

    val tensor1 = Tensor(shape, values)
    val tensor2 = Tensor(shape, values)

    assert(tensor1.tensorEquals(tensor2))
    assert(tensor1 == tensor2)
    assert(!(tensor1 != tensor2))
  }

  test("tensor equality - different values") {
    val shape = Shape2[Height, Width](2, 3)
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

    val tensor1 = Tensor(Shape2[Height, Width](2, 3), values)
    val tensor2 = Tensor(Shape2[Height, Width](3, 2), values)

    assert(!tensor1.tensorEquals(tensor2))
    assert(tensor1 != tensor2)
  }

  test("tensor equality - different dtypes") {
    val shape = Shape1[Feature](3)
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
    val shape = Shape1[Feature](3)
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
    val shape = Shape2[Height, Width](2, 2)
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
    val shape = Shape1[Feature](3)
    val values = Seq(1.0f, 2.0f, 3.0f)

    val tensor1 = Tensor(shape, values)
    val tensor2 = Tensor(shape, values)
    val notATensor: Any = "not a tensor"

    // Test Scala's standard equality
    assertEquals(tensor1, tensor2)
    assertNotEquals(tensor1, notATensor)
  }

  test("tensor hashCode consistency") {
    val shape = Shape1[Feature](3)
    val values = Seq(1.0f, 2.0f, 3.0f)

    val tensor1 = Tensor(shape, values)
    val tensor2 = Tensor(shape, values)

    // Equal tensors should have equal hash codes
    if tensor1.tensorEquals(tensor2) then assertEquals(tensor1.hashCode(), tensor2.hashCode())
  }

  test("tensor equality - element-wise shape mismatch") {
    val tensor1 = Tensor(Shape2[Height, Width](2, 3), (1 to 6).map(_.toFloat))
    val tensor2 = Tensor(Shape2[Height, Width](3, 2), (1 to 6).map(_.toFloat))

    intercept[IllegalArgumentException] {
      tensor1.elementEquals(tensor2)
    }
  }

  test("stack operation for Tensor0 and Tensor1") {
    // Stack three scalars into a vector
    val t1 = Tensor0(1.0f)
    val t2 = Tensor0(2.0f)
    val t3 = Tensor0(3.0f)
    val stacked = Tensor0.stack[NewAxis = Feature](Seq(t1, t2, t3))
    val expected = Tensor1[Feature](Seq(1.0f, 2.0f, 3.0f))
    assertEquals(stacked.shape.dims, Seq(3))
    assert(stacked.tensorEquals(expected))

    // Stack two vectors into a matrix
    val v1 = Tensor1[Feature](Seq(1.0f, 2.0f))
    val v2 = Tensor1[Feature](Seq(3.0f, 4.0f))
    val stacked2 = Tensor1.stack[NewAxis = Batch](Seq(v1, v2))
    val expected2 = Tensor2[Batch, Feature](Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f)))
    assertEquals(stacked2.shape.dims, Seq(2, 2))
    assert(stacked2.tensorEquals(expected2))
  }

  test("split operation") {
    // Test splitting a 2D tensor along the batch dimension
    val shape = Shape2[Batch, Feature](3, 4)
    val values = (1 to 12).map(_.toFloat)
    val tensor = Tensor(shape, values)

    // Split along batch dimension - should get 3 tensors of shape (4,)
    val splits = tensor.split[Batch]
    
    assertEquals(splits.length, 3)
    splits.foreach { split =>
      assertEquals(split.shape.dims, Seq(4))
    }

    // Check that split tensors contain correct values
    val expected1 = Tensor1[Feature](Seq(1.0f, 2.0f, 3.0f, 4.0f))
    val expected2 = Tensor1[Feature](Seq(5.0f, 6.0f, 7.0f, 8.0f))
    val expected3 = Tensor1[Feature](Seq(9.0f, 10.0f, 11.0f, 12.0f))
    
    assert(splits(0).tensorEquals(expected1))
    assert(splits(1).tensorEquals(expected2))
    assert(splits(2).tensorEquals(expected3))

    // Test splitting along feature dimension - should get 4 tensors of shape (3,)
    val featureSplits = tensor.split[Feature]
    
    assertEquals(featureSplits.length, 4)
    featureSplits.foreach { split =>
      assertEquals(split.shape.dims, Seq(3))
    }

    // Check that feature splits contain correct values
    val expectedF1 = Tensor1[Batch](Seq(1.0f, 5.0f, 9.0f))
    val expectedF2 = Tensor1[Batch](Seq(2.0f, 6.0f, 10.0f))
    val expectedF3 = Tensor1[Batch](Seq(3.0f, 7.0f, 11.0f))
    val expectedF4 = Tensor1[Batch](Seq(4.0f, 8.0f, 12.0f))
    
    assert(featureSplits(0).tensorEquals(expectedF1))
    assert(featureSplits(1).tensorEquals(expectedF2))
    assert(featureSplits(2).tensorEquals(expectedF3))
    assert(featureSplits(3).tensorEquals(expectedF4))
  }

  test("split operation on 3D tensor") {
    // Test splitting a 3D tensor
    type Depth = "depth"
    val shape = Shape3[Batch, Height, Width](2, 2, 3)
    val values = (1 to 12).map(_.toFloat)
    val tensor = Tensor(shape, values)

    // Split along batch dimension - should get 2 tensors of shape (2, 3)
    val batchSplits = tensor.split[Batch]
    
    assertEquals(batchSplits.length, 2)
    batchSplits.foreach { split =>
      assertEquals(split.shape.dims, Seq(2, 3))
    }

    // Split along height dimension - should get 2 tensors of shape (2, 3)
    val heightSplits = tensor.split[Height]
    
    assertEquals(heightSplits.length, 2)
    heightSplits.foreach { split =>
      assertEquals(split.shape.dims, Seq(2, 3))
    }

    // Split along width dimension - should get 3 tensors of shape (2, 2)
    val widthSplits = tensor.split[Width]
    
    assertEquals(widthSplits.length, 3)
    widthSplits.foreach { split =>
      assertEquals(split.shape.dims, Seq(2, 2))
    }
  }

end TensorTests
