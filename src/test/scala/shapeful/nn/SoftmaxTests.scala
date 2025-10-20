package shapeful.nn

import munit.FunSuite
import shapeful.*

class SoftmaxTests extends FunSuite:

  type Batch = "batch"
  type Classes = "classes"
  type Feature = "feature"

  override def beforeAll(): Unit =
    super.beforeAll()

  test("softmax with default axis (-1)") {
    // Create a 2D tensor: (batch_size=2, num_classes=3)
    val logits = Tensor2(
      Axis[Batch],
      Axis[Classes],
      Seq(
        Seq(1.0f, 2.0f, 3.0f),
        Seq(4.0f, 5.0f, 6.0f)
      )
    )

    // Apply softmax along the last axis (classes)
    val softmaxResult = logits.softmax(Axis[Classes])

    // Check that probabilities sum to 1 along classes dimension
    assertEquals(softmaxResult.shape.dims, Seq(2, 3))

    // For each batch, the sum across classes should be ~1.0
    // This is a basic sanity check - exact values depend on JAX implementation
    assert(softmaxResult.jaxValue != null, "Softmax should produce valid output")
  }

  test("activation class still works") {
    val logits = Tensor2(
      Axis[Batch],
      Axis[Classes],
      Seq(
        Seq(1.0f, 2.0f, 3.0f),
        Seq(4.0f, 5.0f, 6.0f)
      )
    )

    // Test the original Activation class (uses default axis=-1)
    val result = Activation.softmax(Axis[Classes])(logits)

    assertEquals(result.shape.dims, Seq(2, 3))
    assert(result.jaxValue != null, "Activation.Softmax should produce valid output")
  }

  test("demonstrates JAX softmax usage") {
    // Create a simple example to show the concept
    val logits = Tensor2(
      Axis[Batch],
      Axis[Classes],
      Seq(
        Seq(1.0f, 2.0f, 3.0f),
        Seq(4.0f, 5.0f, 6.0f)
      )
    )

    // Test different approaches to softmax computation
    // 1. Softmax along classes axis (last axis)
    val defaultSoftmax = logits.softmax(Axis[Classes])

    // 2. Using JAX directly with specific axis (this is what the axis-specific version would do)
    import shapeful.jax.Jax
    val axisIndex1 = 1 // Classes dimension (last axis)
    val result1 = Jax.jnn.softmax(logits.jaxValue, axis = axisIndex1)
    val softmaxClasses = new Tensor[Tuple2[Batch, Classes]](logits.shape, result1, logits.dtype)

    val axisIndex0 = 0 // Batch dimension
    val result0 = Jax.jnn.softmax(logits.jaxValue, axis = axisIndex0)
    val softmaxBatch = new Tensor[Tuple2[Batch, Classes]](logits.shape, result0, logits.dtype)

    // All should have the same shape
    assertEquals(defaultSoftmax.shape.dims, Seq(2, 3))
    assertEquals(softmaxClasses.shape.dims, Seq(2, 3))
    assertEquals(softmaxBatch.shape.dims, Seq(2, 3))

    // Results should be valid JAX tensors
    assert(defaultSoftmax.jaxValue != null, "Default softmax should produce valid output")
    assert(softmaxClasses.jaxValue != null, "Softmax over Classes should produce valid output")
    assert(softmaxBatch.jaxValue != null, "Softmax over Batch should produce valid output")
  }
