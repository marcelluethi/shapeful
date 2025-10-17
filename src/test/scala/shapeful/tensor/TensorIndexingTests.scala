package shapeful.tensor

import munit.FunSuite
import shapeful.Label
import shapeful.tensor.Shape.*
import shapeful.tensor.TensorIndexing.*

class TensorIndexingTests extends FunSuite:

  // Test fixtures - define common labels for reuse
  type Height = "height"
  type Width = "width"
  type Batch = "batch"
  type Channel = "channel"
  type Feature = "feature"
  type Index = "index"

  override def beforeAll(): Unit =
    // Initialize Python/JAX environment if needed
    super.beforeAll()

  test("slice - basic slicing along first axis") {
    // Create a 3x4 tensor [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
    val values = Seq(
      Seq(1.0f, 2.0f, 3.0f, 4.0f),
      Seq(5.0f, 6.0f, 7.0f, 8.0f),
      Seq(9.0f, 10.0f, 11.0f, 12.0f)
    )
    val tensor = Tensor2[Height, Width](values)

    // Slice first 2 rows (indices 0 and 1)
    val sliced = tensor.slice[Height](0, 2)

    assertEquals(sliced.shape.dims, Seq(2, 4))
    assertEquals(sliced.shape.dim[Height], 2)
    assertEquals(sliced.shape.dim[Width], 4)
  }

  test("slice - basic slicing along second axis") {
    // Create a 3x4 tensor
    val values = Seq(
      Seq(1.0f, 2.0f, 3.0f, 4.0f),
      Seq(5.0f, 6.0f, 7.0f, 8.0f),
      Seq(9.0f, 10.0f, 11.0f, 12.0f)
    )
    val tensor = Tensor2[Height, Width](values)

    // Slice first 3 columns (indices 0, 1, 2)
    val sliced = tensor.slice[Width](0, 3)

    assertEquals(sliced.shape.dims, Seq(3, 3))
    assertEquals(sliced.shape.dim[Height], 3)
    assertEquals(sliced.shape.dim[Width], 3)
  }

  test("slice - middle slice") {
    // Create a 4x4 tensor
    val values = Seq(
      Seq(1.0f, 2.0f, 3.0f, 4.0f),
      Seq(5.0f, 6.0f, 7.0f, 8.0f),
      Seq(9.0f, 10.0f, 11.0f, 12.0f),
      Seq(13.0f, 14.0f, 15.0f, 16.0f)
    )
    val tensor = Tensor2[Height, Width](values)

    // Slice rows 1 and 2 (indices 1, 2)
    val sliced = tensor.slice[Height](1, 3)

    assertEquals(sliced.shape.dims, Seq(2, 4))
    assertEquals(sliced.shape.dim[Height], 2)
    assertEquals(sliced.shape.dim[Width], 4)
  }

  test("slice - single element slice") {
    val values = Seq(
      Seq(1.0f, 2.0f, 3.0f),
      Seq(4.0f, 5.0f, 6.0f)
    )
    val tensor = Tensor2[Height, Width](values)

    // Slice just one row
    val sliced = tensor.slice[Height](0, 1)

    assertEquals(sliced.shape.dims, Seq(1, 3))
    assertEquals(sliced.shape.dim[Height], 1)
    assertEquals(sliced.shape.dim[Width], 3)
  }

  test("slice - empty slice") {
    val values = Seq(
      Seq(1.0f, 2.0f, 3.0f),
      Seq(4.0f, 5.0f, 6.0f)
    )
    val tensor = Tensor2[Height, Width](values)

    // Empty slice (start == end)
    val sliced = tensor.slice[Height](1, 1)

    assertEquals(sliced.shape.dims, Seq(0, 3))
    assertEquals(sliced.shape.dim[Height], 0)
    assertEquals(sliced.shape.dim[Width], 3)
  }

  test("slice - boundary validation - negative start") {
    val values = Seq(1.0f, 2.0f, 3.0f)
    val tensor = Tensor1[Feature](values)

    intercept[IllegalArgumentException] {
      tensor.slice[Feature](-1, 2)
    }
  }

  test("slice - boundary validation - start > axis size") {
    val values = Seq(1.0f, 2.0f, 3.0f)
    val tensor = Tensor1[Feature](values)

    intercept[IllegalArgumentException] {
      tensor.slice[Feature](4, 5)
    }
  }

  test("slice - boundary validation - end < start") {
    val values = Seq(1.0f, 2.0f, 3.0f)
    val tensor = Tensor1[Feature](values)

    intercept[IllegalArgumentException] {
      tensor.slice[Feature](2, 1)
    }
  }

  test("slice - boundary validation - end > axis size") {
    val values = Seq(1.0f, 2.0f, 3.0f)
    val tensor = Tensor1[Feature](values)

    intercept[IllegalArgumentException] {
      tensor.slice[Feature](0, 4)
    }
  }

  test("sliceWithSize - convenience method") {
    val values = Seq(
      Seq(1.0f, 2.0f, 3.0f, 4.0f),
      Seq(5.0f, 6.0f, 7.0f, 8.0f),
      Seq(9.0f, 10.0f, 11.0f, 12.0f)
    )
    val tensor = Tensor2[Height, Width](values)

    // Slice 2 elements starting from index 1
    val sliced = tensor.sliceWithSize[Height](1, 2)

    assertEquals(sliced.shape.dims, Seq(2, 4))
    assertEquals(sliced.shape.dim[Height], 2)
    assertEquals(sliced.shape.dim[Width], 4)
  }

  test("gather - basic gathering with indices") {
    // Create a 4x3 tensor
    val values = Seq(
      Seq(1.0f, 2.0f, 3.0f),
      Seq(4.0f, 5.0f, 6.0f),
      Seq(7.0f, 8.0f, 9.0f),
      Seq(10.0f, 11.0f, 12.0f)
    )
    val tensor = Tensor2[Height, Width](values)

    // Gather rows at indices [0, 2, 1] - using integers directly
    val indices = Tensor1[Index](Seq(0, 2, 1).map(_.toFloat))
    val gathered = tensor.gather[Height, Index](indices)

    assertEquals(gathered.shape.dims, Seq(3, 3))
    assertEquals(gathered.shape.dim[Height], 3) // 3 indices
    assertEquals(gathered.shape.dim[Width], 3)
  }

  test("gather - gathering along second axis") {
    // Create a 3x4 tensor
    val values = Seq(
      Seq(1.0f, 2.0f, 3.0f, 4.0f),
      Seq(5.0f, 6.0f, 7.0f, 8.0f),
      Seq(9.0f, 10.0f, 11.0f, 12.0f)
    )
    val tensor = Tensor2[Height, Width](values)

    // Gather columns at indices [1, 3, 0] - using integers directly
    val indices = Tensor1[Index](Seq(1, 3, 0).map(_.toFloat))
    val gathered = tensor.gather[Width, Index](indices)

    assertEquals(gathered.shape.dims, Seq(3, 3))
    assertEquals(gathered.shape.dim[Height], 3)
    assertEquals(gathered.shape.dim[Width], 3) // 3 indices
  }

  test("gather - single index") {
    val values = Seq(
      Seq(1.0f, 2.0f, 3.0f),
      Seq(4.0f, 5.0f, 6.0f),
      Seq(7.0f, 8.0f, 9.0f)
    )
    val tensor = Tensor2[Height, Width](values)

    // Gather just one row (index 1)
    val indices = Tensor1[Index](Seq(1).map(_.toFloat))
    val gathered = tensor.gather[Height, Index](indices)

    assertEquals(gathered.shape.dims, Seq(1, 3))
    assertEquals(gathered.shape.dim[Height], 1)
    assertEquals(gathered.shape.dim[Width], 3)
  }

  test("gather - duplicate indices") {
    val values = Seq(
      Seq(1.0f, 2.0f),
      Seq(3.0f, 4.0f),
      Seq(5.0f, 6.0f)
    )
    val tensor = Tensor2[Height, Width](values)

    // Gather with duplicate indices [0, 0, 2, 0]
    val indices = Tensor1[Index](Seq(0, 0, 2, 0).map(_.toFloat))
    val gathered = tensor.gather[Height, Index](indices)

    assertEquals(gathered.shape.dims, Seq(4, 2))
    assertEquals(gathered.shape.dim[Height], 4) // 4 indices
    assertEquals(gathered.shape.dim[Width], 2)
  }

  test("gather - boundary validation - negative indices") {
    val values = Seq(1.0f, 2.0f, 3.0f)
    val tensor = Tensor1[Feature](values)

    val indices = Tensor1[Index](Seq(-1, 1).map(_.toFloat))

    intercept[IllegalArgumentException] {
      tensor.gather[Feature, Index](indices)
    }
  }

  test("gather - boundary validation - indices out of bounds") {
    val values = Seq(1.0f, 2.0f, 3.0f)
    val tensor = Tensor1[Feature](values)

    val indices = Tensor1[Index](Seq(0, 3).map(_.toFloat)) // 3 is out of bounds for size 3

    intercept[IllegalArgumentException] {
      tensor.gather[Feature, Index](indices)
    }
  }

  test("gatherSeq - convenience method with sequence") {
    val values = Seq(
      Seq(1.0f, 2.0f, 3.0f),
      Seq(4.0f, 5.0f, 6.0f),
      Seq(7.0f, 8.0f, 9.0f),
      Seq(10.0f, 11.0f, 12.0f)
    )
    val tensor = Tensor2[Height, Width](values)

    // Gather rows using sequence of integers
    val gathered = tensor.gatherSeq[Height](Seq(3, 0, 1))

    assertEquals(gathered.shape.dims, Seq(3, 3))
    assertEquals(gathered.shape.dim[Height], 3)
    assertEquals(gathered.shape.dim[Width], 3)
  }

  test("gather - 1D tensor") {
    val values = Seq(10.0f, 20.0f, 30.0f, 40.0f, 50.0f)
    val tensor = Tensor1[Feature](values)

    val indices = Tensor1[Index](Seq(4, 0, 2).map(_.toFloat))
    val gathered = tensor.gather[Feature, Index](indices)

    assertEquals(gathered.shape.dims, Seq(3))
    assertEquals(gathered.shape.dim[Feature], 3)
  }

  test("slice - 3D tensor") {
    // Create a 2x3x4 tensor using Shape3
    val shape = Shape(Axis[Batch] -> 2, Axis[Height] -> 3, Axis[Width] -> 4)
    val values = (1 to 24).map(_.toFloat)
    val tensor = Tensor(shape, values, DType.Float32)

    // Slice along the batch dimension
    val sliced = tensor.slice[Batch](0, 1)

    assertEquals(sliced.shape.dims, Seq(1, 3, 4))
    assertEquals(sliced.shape.dim[Batch], 1)
    assertEquals(sliced.shape.dim[Height], 3)
    assertEquals(sliced.shape.dim[Width], 4)
  }

  test("gather - 3D tensor") {
    // Create a 2x3x4 tensor
    val shape = Shape(Axis[Batch] -> 2, Axis[Height] -> 3, Axis[Width] -> 4)
    val values = (1 to 24).map(_.toFloat)
    val tensor = Tensor(shape, values, DType.Float32)

    // Gather along height dimension
    val indices = Tensor1[Index](Seq(2, 0).map(_.toFloat))
    val gathered = tensor.gather[Height, Index](indices)

    assertEquals(gathered.shape.dims, Seq(2, 2, 4))
    assertEquals(gathered.shape.dim[Batch], 2)
    assertEquals(gathered.shape.dim[Height], 2) // 2 indices
    assertEquals(gathered.shape.dim[Width], 4)
  }

end TensorIndexingTests
