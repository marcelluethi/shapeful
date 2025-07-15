package shapeful.jax

import scala.language.experimental.namedTypeArguments

import munit.FunSuite

import shapeful.*
import shapeful.jax.Jax
import shapeful.autodiff.ToPyTree
import shapeful.tensor.DType

class PyTreeTests extends FunSuite:

  // Test labels
  type Feature = "feature"
  type Batch = "batch"
  type Hidden = "hidden"

  override def beforeAll(): Unit =
    // Initialize Python/JAX environment if needed
    super.beforeAll()

  test("single tensor can be converted to PyTree and recovered") {
    val original = Tensor1[Feature](Seq(1.0f, 2.0f, 3.0f, 4.0f))
    val converter = summon[ToPyTree[Tensor1[Feature]]]

    // Convert to PyTree
    val pyTree = converter.toPyTree(original)
    assert(pyTree != null, "PyTree conversion should not be null")

    // Convert back from PyTree
    val recovered = converter.fromPyTree(pyTree)

    // Check that values match
    assert(
      recovered.approxEquals(original, tolerance = 1e-6f),
      s"Recovered tensor should match original: original=${original.toString}, recovered=${recovered.toString}"
    )

    // Check that shapes match
    assert(
      recovered.shape.dims == original.shape.dims,
      s"Shapes should match: original=${original.shape.dims}, recovered=${recovered.shape.dims}"
    )
  }

  test("scalar tensor can be converted to PyTree and recovered") {
    val original = Tensor0(42.0f)
    val converter = summon[ToPyTree[Tensor0]]

    // Convert to PyTree
    val pyTree = converter.toPyTree(original)
    assert(pyTree != null, "PyTree conversion should not be null")

    // Convert back from PyTree
    val recovered = converter.fromPyTree(pyTree)

    // Check that values match
    assert(
      recovered.approxEquals(original, tolerance = 1e-6f),
      s"Recovered tensor should match original: ${original.toFloat} vs ${recovered.toFloat}"
    )
  }

  test("2D tensor can be converted to PyTree and recovered") {
    val original = Tensor2[Batch, Feature](
      Seq(
        Seq(1.0f, 2.0f, 3.0f),
        Seq(4.0f, 5.0f, 6.0f)
      )
    )
    val converter = summon[ToPyTree[Tensor2[Batch, Feature]]]

    // Convert to PyTree
    val pyTree = converter.toPyTree(original)
    assert(pyTree != null, "PyTree conversion should not be null")

    // Convert back from PyTree
    val recovered = converter.fromPyTree(pyTree)

    // Check that values match
    assert(recovered.approxEquals(original, tolerance = 1e-6f), s"Recovered tensor should match original")

    // Check that shapes match
    assert(
      recovered.shape.dims == original.shape.dims,
      s"Shapes should match: original=${original.shape.dims}, recovered=${recovered.shape.dims}"
    )
  }

  test("tuple of 2 tensors can be converted to PyTree and recovered") {
    val tensor1 = Tensor1[Feature](Seq(1.0f, 2.0f, 3.0f))
    val tensor2 = Tensor2[Batch, Hidden](
      Seq(
        Seq(4.0f, 5.0f),
        Seq(6.0f, 7.0f),
        Seq(8.0f, 9.0f)
      )
    )
    val original = (tensor1, tensor2)
    val converter = summon[ToPyTree[(Tensor1[Feature], Tensor2[Batch, Hidden])]]

    // Convert to PyTree
    val pyTree = converter.toPyTree(original)
    assert(pyTree != null, "PyTree conversion should not be null")

    // Convert back from PyTree
    val recovered = converter.fromPyTree(pyTree)

    // Check that both tensors match
    assert(recovered._1.approxEquals(original._1, tolerance = 1e-6f), s"First tensor should match original")
    assert(recovered._2.approxEquals(original._2, tolerance = 1e-6f), s"Second tensor should match original")

    // Check that shapes match
    assert(recovered._1.shape.dims == original._1.shape.dims, s"First tensor shape should match")
    assert(recovered._2.shape.dims == original._2.shape.dims, s"Second tensor shape should match")
  }

  test("tuple of 3 tensors can be converted to PyTree and recovered") {
    val tensor1 = Tensor0(1.0f)
    val tensor2 = Tensor1[Feature](Seq(2.0f, 3.0f))
    val tensor3 = Tensor2[Batch, Hidden](
      Seq(
        Seq(4.0f, 5.0f),
        Seq(6.0f, 7.0f)
      )
    )
    val original = (tensor1, tensor2, tensor3)
    val converter = summon[ToPyTree[(Tensor0, Tensor1[Feature], Tensor2[Batch, Hidden])]]

    // Convert to PyTree
    val pyTree = converter.toPyTree(original)
    assert(pyTree != null, "PyTree conversion should not be null")

    // Convert back from PyTree
    val recovered = converter.fromPyTree(pyTree)

    // Check that all tensors match
    assert(recovered._1.approxEquals(original._1, tolerance = 1e-6f), s"First tensor should match original")
    assert(recovered._2.approxEquals(original._2, tolerance = 1e-6f), s"Second tensor should match original")
    assert(recovered._3.approxEquals(original._3, tolerance = 1e-6f), s"Third tensor should match original")

    // Check that shapes match
    assert(recovered._1.shape.dims == original._1.shape.dims, s"First tensor shape should match")
    assert(recovered._2.shape.dims == original._2.shape.dims, s"Second tensor shape should match")
    assert(recovered._3.shape.dims == original._3.shape.dims, s"Third tensor shape should match")
  }

  test("dtype preservation in PyTree conversion") {
    // Test that dtype is correctly extracted and preserved
    val originalFloat32 = Tensor1[Feature](Seq(1.0f, 2.0f, 3.0f))
    val converter = summon[ToPyTree[Tensor1[Feature]]]

    val pyTree = converter.toPyTree(originalFloat32)
    val recovered = converter.fromPyTree(pyTree)

    // Note: The current implementation always uses Float32, but this test
    // is here for when dtype preservation is fully implemented
    assert(recovered.dtype == DType.Float32, s"Dtype should be preserved: expected Float32, got ${recovered.dtype}")
  }

end PyTreeTests
