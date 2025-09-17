package shapeful.nn

import scala.language.experimental.namedTypeArguments
import munit.FunSuite
import shapeful.*
import shapeful.tensor.Shape.*
import shapeful.nn.Layer.LayerDim

class LinearLayerTests extends FunSuite:

  // Test fixtures - define common labels for reuse
  type InputDim = "input"
  type OutputDim = "output"
  type HiddenDim = "hidden"

  override def beforeAll(): Unit =
    // Initialize Python/JAX environment if needed
    super.beforeAll()

  test("Linear layer forward pass transforms input correctly") {
    val inputDim = 3
    val outputDim = 2

    // Create a linear layer
    val layer = Linear[InputDim, OutputDim]()

    // Create specific parameters (not random for deterministic test)
    val params = Linear.Params[InputDim, OutputDim](
      weight = Tensor(
        Shape2[InputDim, OutputDim](inputDim, outputDim),
        Seq(1.0f, 2.0f, 0.5f, -1.0f, 0.0f, 1.5f), // 3x2 matrix
        DType.Float32
      ),
      bias = Tensor(
        Shape1[OutputDim](outputDim),
        Seq(0.1f, -0.2f), // bias vector
        DType.Float32
      )
    )

    // Create input tensor
    val input = Tensor(
      Shape1[InputDim](inputDim),
      Seq(1.0f, 2.0f, 3.0f), // input vector
      DType.Float32
    )

    // Apply forward pass
    val forward = layer(params)
    val output = forward(input)

    // Expected output: [1, 2, 3] * [[1, 2], [0.5, -1], [0, 1.5]] + [0.1, -0.2]
    // = [1*1 + 2*0.5 + 3*0, 1*2 + 2*(-1) + 3*1.5] + [0.1, -0.2]
    // = [1 + 1 + 0, 2 - 2 + 4.5] + [0.1, -0.2]
    // = [2, 4.5] + [0.1, -0.2] = [2.1, 4.3]

    assertEquals(output.shape.dims, Seq(outputDim))

    // Create expected output tensor for comparison
    val expectedOutput = Tensor(
      Shape1[OutputDim](outputDim),
      Seq(2.1f, 4.3f),
      DType.Float32
    )

    // Check the computed values with some tolerance for floating point
    assert(
      output.approxEquals(expectedOutput, tolerance = 1e-5f),
      s"Expected ${expectedOutput}, got ${output}"
    )
  }

  test("Xavier initialization produces reasonable parameter ranges") {
    val inputDim = 100
    given LayerDim[InputDim] = LayerDim(inputDim)
    val outputDim = 50
    given LayerDim[OutputDim] = LayerDim(outputDim)
    val key = shapeful.random.Random.Key(42)

    val params = Linear.xavier[InputDim, OutputDim](key)

    // Check weight dimensions
    assertEquals(params.weight.shape.dims, Seq(inputDim, outputDim))

    // Check bias dimensions (should be zeros)
    assertEquals(params.bias.shape.dims, Seq(outputDim))

    // Check that bias is actually zeros
    val expectedBias = Tensor.zeros(Shape1[OutputDim](outputDim))
    assert(
      params.bias.tensorEquals(expectedBias),
      "Bias should be initialized to zeros"
    )

    // Check that weights are not all the same (basic randomness check)
    // We'll create two different parameter sets and ensure they're different
    val key2 = shapeful.random.Random.Key(123)
    val params2 = Linear.xavier[InputDim, OutputDim](key2)
    assert(
      !params.weight.tensorEquals(params2.weight),
      "Weights should be randomly initialized (different instances should be different)"
    )
  }

  test("He initialization produces reasonable parameter ranges") {
    val inputDim = 100
    val outputDim = 50
    given LayerDim[InputDim] = LayerDim(inputDim)
    given LayerDim[OutputDim] = LayerDim(outputDim)
    val key = shapeful.random.Random.Key(456)

    val params = Linear.he[InputDim, OutputDim](key)

    // Check weight dimensions
    assertEquals(params.weight.shape.dims, Seq(inputDim, outputDim))

    // Check bias dimensions (should be zeros)
    assertEquals(params.bias.shape.dims, Seq(outputDim))

    // Check that bias is actually zeros
    val expectedBias = Tensor.zeros(Shape1[OutputDim](outputDim))
    assert(
      params.bias.tensorEquals(expectedBias),
      "Bias should be initialized to zeros"
    )

    // Check that weights are not all the same (basic randomness check)
    // We'll create two different parameter sets and ensure they're different
    val key2 = shapeful.random.Random.Key(789)
    val params2 = Linear.he[InputDim, OutputDim](key2)
    assert(
      !params.weight.tensorEquals(params2.weight),
      "Weights should be randomly initialized (different instances should be different)"
    )
  }
