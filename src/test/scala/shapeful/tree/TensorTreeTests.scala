package shapeful.tree

import scala.language.experimental.namedTypeArguments
import munit.FunSuite
import shapeful.*
import shapeful.autodiff.*

class TensorTreeTests extends FunSuite:

  // Import tensor operations locally to avoid conflicts
  import shapeful.tensor.TensorOps.*

  // Test fixtures - define common labels for reuse
  type Feature = "feature"
  type Hidden = "hidden"
  type Output = "output"
  type Batch = "batch"

  override def beforeAll(): Unit =
    // Initialize Python/JAX environment if needed
    super.beforeAll()

  // Test parameter structures for testing
  case class SimpleParams(value: Tensor0) derives TensorTree

  case class LinearParams(
      weight: Tensor2[Feature, Hidden],
      bias: Tensor1[Hidden]
  ) derives TensorTree

  case class NetworkParams(
      layer1: LinearParams,
      layer2: LinearParams,
      scale: Tensor0
  ) derives TensorTree

  // Basic TensorTree functionality tests
  test("TensorTree works with single tensors") {
    val original = Tensor0(5.0f)
    val doubled = TensorTree[Tensor0].map(original, [T <: Tuple] => (t: Tensor[T]) => (t * Tensor0(2.0f)))

    val expected = Tensor0(10.0f)
    assert(doubled.approxEquals(expected, tolerance = 1e-5f), s"Expected ${expected.toFloat}, got ${doubled.toFloat}")
  }

  test("TensorTree map preserves structure for simple params") {
    val params = SimpleParams(Tensor0(3.0f))
    val scaled = TensorTree[SimpleParams].map(params, [T <: Tuple] => (t: Tensor[T]) => (t * Tensor0(2.0f)))

    val expected = Tensor0(6.0f)
    assert(
      scaled.value.approxEquals(expected, tolerance = 1e-5f),
      s"Expected ${expected.toFloat}, got ${scaled.value.toFloat}"
    )
  }

  test("TensorTree map works with linear layer parameters") {
    val params = LinearParams(
      weight = Tensor2[Feature, Hidden](Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f))),
      bias = Tensor1[Hidden](Seq(0.5f, 1.0f))
    )

    val doubled = TensorTree[LinearParams].map(params, [T <: Tuple] => (t: Tensor[T]) => (t * Tensor0(2.0f)))

    val expectedWeight = Tensor2[Feature, Hidden](Seq(Seq(2.0f, 4.0f), Seq(6.0f, 8.0f)))
    val expectedBias = Tensor1[Hidden](Seq(1.0f, 2.0f))

    assert(doubled.weight.approxEquals(expectedWeight, tolerance = 1e-5f), "Weight should be doubled")
    assert(doubled.bias.approxEquals(expectedBias, tolerance = 1e-5f), "Bias should be doubled")
  }

  test("TensorTree map handles nested structures") {
    val params = NetworkParams(
      layer1 = LinearParams(
        weight = Tensor2[Feature, Hidden](Seq(Seq(1.0f, 2.0f))),
        bias = Tensor1[Hidden](Seq(0.5f, 1.0f))
      ),
      layer2 = LinearParams(
        weight = Tensor2[Feature, Hidden](Seq(Seq(3.0f, 4.0f))),
        bias = Tensor1[Hidden](Seq(1.5f, 2.0f))
      ),
      scale = Tensor0(2.0f)
    )

    val halved = TensorTree[NetworkParams].map(params, [T <: Tuple] => (t: Tensor[T]) => (t / Tensor0(2.0f)))

    // Check that all tensors were halved
    assert(
      halved.layer1.weight.at((0, 0)).get.approxEquals(Tensor0(0.5f), tolerance = 1e-5f),
      "Layer1 weight should be halved"
    )
    assert(
      halved.layer1.bias.at(Tuple1(0)).get.approxEquals(Tensor0(0.25f), tolerance = 1e-5f),
      "Layer1 bias should be halved"
    )
    assert(
      halved.layer2.weight.at((0, 0)).get.approxEquals(Tensor0(1.5f), tolerance = 1e-5f),
      "Layer2 weight should be halved"
    )
    assert(halved.scale.approxEquals(Tensor0(1.0f), tolerance = 1e-5f), "Scale should be halved")
  }

  test("TensorTree zipMap combines two simple tensors") {
    val t1 = Tensor0(3.0f)
    val t2 = Tensor0(4.0f)

    val sum = TensorTree[Tensor0].zipMap(t1, t2, [T <: Tuple] => (a: Tensor[T], b: Tensor[T]) => (a + b))

    val expected = Tensor0(7.0f)
    assert(sum.approxEquals(expected, tolerance = 1e-5f), s"Expected ${expected.toFloat}, got ${sum.toFloat}")
  }

  test("TensorTree zipMap works with simple parameter structures") {
    val params1 = SimpleParams(Tensor0(2.0f))
    val params2 = SimpleParams(Tensor0(3.0f))

    val combined = TensorTree[SimpleParams].zipMap(
      params1,
      params2,
      [T <: Tuple] => (a: Tensor[T], b: Tensor[T]) => (a * b)
    )

    val expected = Tensor0(6.0f)
    assert(
      combined.value.approxEquals(expected, tolerance = 1e-5f),
      s"Expected ${expected.toFloat}, got ${combined.value.toFloat}"
    )
  }

  test("TensorTree zipMap combines linear layer parameters") {
    val params1 = LinearParams(
      weight = Tensor2[Feature, Hidden](Seq(Seq(1.0f, 2.0f), Seq(3.0f, 4.0f))),
      bias = Tensor1[Hidden](Seq(1.0f, 2.0f))
    )
    val params2 = LinearParams(
      weight = Tensor2[Feature, Hidden](Seq(Seq(2.0f, 3.0f), Seq(4.0f, 5.0f))),
      bias = Tensor1[Hidden](Seq(0.5f, 1.5f))
    )

    val sum = TensorTree[LinearParams].zipMap(
      params1,
      params2,
      [T <: Tuple] => (a: Tensor[T], b: Tensor[T]) => (a + b)
    )

    val expectedWeight = Tensor2[Feature, Hidden](Seq(Seq(3.0f, 5.0f), Seq(7.0f, 9.0f)))
    val expectedBias = Tensor1[Hidden](Seq(1.5f, 3.5f))

    assert(sum.weight.approxEquals(expectedWeight, tolerance = 1e-5f), "Weight should be summed")
    assert(sum.bias.approxEquals(expectedBias, tolerance = 1e-5f), "Bias should be summed")
  }

  test("TensorTree zipMap handles nested structures") {
    val params1 = NetworkParams(
      layer1 = LinearParams(
        weight = Tensor2[Feature, Hidden](Seq(Seq(1.0f, 2.0f))),
        bias = Tensor1[Hidden](Seq(1.0f, 2.0f))
      ),
      layer2 = LinearParams(
        weight = Tensor2[Feature, Hidden](Seq(Seq(3.0f, 4.0f))),
        bias = Tensor1[Hidden](Seq(3.0f, 4.0f))
      ),
      scale = Tensor0(1.0f)
    )

    val params2 = NetworkParams(
      layer1 = LinearParams(
        weight = Tensor2[Feature, Hidden](Seq(Seq(2.0f, 3.5f))),
        bias = Tensor1[Hidden](Seq(0.5f, 1.5f))
      ),
      layer2 = LinearParams(
        weight = Tensor2[Feature, Hidden](Seq(Seq(5.0f, 4.0f))),
        bias = Tensor1[Hidden](Seq(2.0f, 3.0f))
      ),
      scale = Tensor0(2.0f)
    )

    val difference = TensorTree[NetworkParams].zipMap(
      params1,
      params2,
      [T <: Tuple] => (a: Tensor[T], b: Tensor[T]) => (a - b)
    )
    val expectedWeightDiffLayer2 =
      Tensor2[Feature, Hidden](Seq(Seq(-2f, 0f)))
    val expectedBiasDiffLayer2 =
      Tensor1[Hidden](Seq(1f, 1f))

    assert(
      difference.layer2.weight.approxEquals(expectedWeightDiffLayer2),
      "Layer2 weight difference should be correct"
    )
  }

  test("extension methods work correctly") {
    val params = SimpleParams(Tensor0(4.0f))
    val scaled = params.map([T <: Tuple] => (t: Tensor[T]) => (t * Tensor0(3.0f)))

    val expected = Tensor0(12.0f)
    assert(
      scaled.value.approxEquals(expected, tolerance = 1e-5f),
      s"Extension method should work: expected ${expected.toFloat}, got ${scaled.value.toFloat}"
    )
  }

  test("extension zipMap method works correctly") {
    val params1 = SimpleParams(Tensor0(6.0f))
    val params2 = SimpleParams(Tensor0(4.0f))

    val ratio = params1.zipMap(params2, [T <: Tuple] => (a: Tensor[T], b: Tensor[T]) => (a / b))

    val expected = Tensor0(1.5f)
    assert(
      ratio.value.approxEquals(expected, tolerance = 1e-5f),
      s"Extension zipMap should work: expected ${expected.toFloat}, got ${ratio.value.toFloat}"
    )
  }

  test("TensorTree.apply summon syntax works") {
    val params = SimpleParams(Tensor0(7.0f))
    val tree = TensorTree[SimpleParams]
    val negated = tree.map(params, [T <: Tuple] => (t: Tensor[T]) => (t * Tensor0(-1.0f)))

    val expected = Tensor0(-7.0f)
    assert(
      negated.value.approxEquals(expected, tolerance = 1e-5f),
      s"TensorTree.apply should work: expected ${expected.toFloat}, got ${negated.value.toFloat}"
    )
  }

  test("multiple operations can be chained") {
    val params = LinearParams(
      weight = Tensor2[Feature, Hidden](Seq(Seq(2.0f, 4.0f))),
      bias = Tensor1[Hidden](Seq(1.0f, 2.0f))
    )

    // Chain multiple operations: double, then add bias offset, then square
    val step1 = params.map([T <: Tuple] => (t: Tensor[T]) => (t * Tensor0(2.0f)))
    val step2 = step1.map([T <: Tuple] => (t: Tensor[T]) => (t + Tensor0(1.0f)))
    val step3 = step2.map([T <: Tuple] => (t: Tensor[T]) => (t * t))

    // weight: 2.0 -> 4.0 -> 5.0 -> 25.0
    // bias: 1.0 -> 2.0 -> 3.0 -> 9.0
    assert(
      step3.weight.at((0, 0)).get.approxEquals(Tensor0(25.0f), tolerance = 1e-5f),
      "Chained operations should work correctly"
    )
    assert(
      step3.bias.at(Tuple1(0)).get.approxEquals(Tensor0(9.0f), tolerance = 1e-5f),
      "Chained operations should work correctly"
    )
  }

  test("zipMap with different operations for different tensor types") {
    val params1 = LinearParams(
      weight = Tensor2[Feature, Hidden](Seq(Seq(8.0f, 12.0f))),
      bias = Tensor1[Hidden](Seq(4.0f, 6.0f))
    )
    val params2 = LinearParams(
      weight = Tensor2[Feature, Hidden](Seq(Seq(2.0f, 3.0f))),
      bias = Tensor1[Hidden](Seq(2.0f, 3.0f))
    )

    // Divide weights, subtract biases
    val weightDivided = TensorTree[LinearParams].zipMap(
      params1,
      params2,
      [T <: Tuple] => (a: Tensor[T], b: Tensor[T]) => (a / b)
    )

    // weight: 8/2=4, 12/3=4
    // bias: 4/2=2, 6/3=2
    assert(
      weightDivided.weight.at((0, 0)).get.approxEquals(Tensor0(4.0f), tolerance = 1e-5f),
      "Weight division should be correct"
    )
    assert(
      weightDivided.weight.at((0, 1)).get.approxEquals(Tensor0(4.0f), tolerance = 1e-5f),
      "Weight division should be correct"
    )
    assert(
      weightDivided.bias.at(Tuple1(0)).get.approxEquals(Tensor0(2.0f), tolerance = 1e-5f),
      "Bias division should be correct"
    )
    assert(
      weightDivided.bias.at(Tuple1(1)).get.approxEquals(Tensor0(2.0f), tolerance = 1e-5f),
      "Bias division should be correct"
    )
  }

end TensorTreeTests
