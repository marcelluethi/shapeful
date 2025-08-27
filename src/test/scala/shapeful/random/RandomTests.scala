package shapeful.random

import scala.language.experimental.namedTypeArguments
import munit.FunSuite
import shapeful.*
import shapeful.tensor.Shape.*
import shapeful.random.Random
import math.{abs, sqrt}

class RandomTests extends FunSuite:

  // Type aliases for test dimensions
  type Sample = "sample"
  type Feature = "feature"
  type Batch = "batch"

  // Test tolerance for floating point comparisons
  val tolerance = 1e-5f

  def assertApproxEqual(actual: Float, expected: Float, tolerance: Float = tolerance): Unit =
    if abs(actual - expected) > tolerance then fail(s"Expected $expected, but got $actual (tolerance: $tolerance)")

  override def beforeAll(): Unit =
    super.beforeAll()

  test("Random.Key creation and basic operations") {
    val key1 = Random.Key(42)
    val key2 = Random.Key(42)
    val key3 = Random.Key(123)

    // Keys with same seed should be deterministic but different objects
    assert(key1 != key2, "Keys should be different objects even with same seed")

    // Key splitting should work
    val (splitKey1, splitKey2) = key1.split2()
    assert(splitKey1 != splitKey2, "Split keys should be different")

    val keys = key1.split(5)
    assert(keys.length == 5, "Should split into requested number of keys")

    // All split keys should be different
    for i <- keys.indices do
      for j <- (i + 1) until keys.length do assert(keys(i) != keys(j), s"Split keys $i and $j should be different")
  }

  test("Random number generation consistency") {
    val key = Random.Key(42)

    // Same key should produce same random numbers
    val rand1 = Random.uniform(Shape0, key)
    val rand2 = Random.uniform(Shape0, key)

    assertApproxEqual(rand1.toFloat, rand2.toFloat, 1e-7f)
  }

  test("Random number generation with different keys") {
    val key1 = Random.Key(42)
    val key2 = Random.Key(123)

    val rand1 = Random.uniform(Shape0, key1)
    val rand2 = Random.uniform(Shape0, key2)

    // Different keys should (very likely) produce different numbers
    assert(abs(rand1.toFloat - rand2.toFloat) > tolerance, "Different keys should produce different random numbers")
  }

  test("vmapSample basic functionality") {
    val baseKey = Random.Key(42)
    val numSamples = 5

    // Test vmapSample with scalar output
    val scalarSamples = Random.vmapSample(baseKey, numSamples, (key: Random.Key) => Random.uniform(Shape0, key))

    // Check output shape - should be (Sample,)
    assert(scalarSamples.shape.dims.length == 1, "vmapSample output should have Sample dimension")
    assert(scalarSamples.shape.dims(0) == numSamples, s"Should have $numSamples samples")

    // Check that samples are different (very high probability)
    // vmapSample should return Tensor1[Sample] when input function returns Tensor0
    val mean = scalarSamples.mean.toFloat
    val variance = scalarSamples.variance.toFloat

    // With uniform random samples, variance should be > 0
    assert(variance > 1e-6f, s"Samples should have variance: $variance")
  }

  test("vmapSample with vector output") {
    val baseKey = Random.Key(123)
    val numSamples = 3
    val featureSize = 4

    // Test vmapSample with vector output
    val vectorSamples =
      Random.vmapSample(baseKey, numSamples, (key: Random.Key) => Random.uniform(Shape1[Feature](featureSize), key))

    // Check output shape - should be (Sample, Feature)
    assert(vectorSamples.shape.dims.length == 2, "vmapSample output should have Sample and Feature dimensions")
    assert(vectorSamples.shape.dims(0) == numSamples, s"Should have $numSamples samples")
    assert(vectorSamples.shape.dims(1) == featureSize, s"Should have $featureSize features")

    // Check that samples are different
    // For now, just check that we got the right shape and some variance
    val sampleVariance = vectorSamples.variance.toFloat
    assert(sampleVariance > tolerance, "Vector samples should have variance")
  }

  test("vmapSample deterministic with same base key") {
    val baseKey = Random.Key(456)
    val numSamples = 4

    // Generate samples twice with same key
    val samples1 = Random.vmapSample(baseKey, numSamples, (key: Random.Key) => Random.uniform(Shape0, key))

    val samples2 = Random.vmapSample(baseKey, numSamples, (key: Random.Key) => Random.uniform(Shape0, key))

    // Should be identical
    assert(samples1.approxEquals(samples2, tolerance), "vmapSample should be deterministic with same base key")
  }

  test("vmapSample different results with different base keys") {
    val baseKey1 = Random.Key(789)
    val baseKey2 = Random.Key(987)
    val numSamples = 3

    val samples1 = Random.vmapSample(baseKey1, numSamples, (key: Random.Key) => Random.uniform(Shape0, key))

    val samples2 = Random.vmapSample(baseKey2, numSamples, (key: Random.Key) => Random.uniform(Shape0, key))

    // Should be different (very high probability)
    assert(
      !samples1.approxEquals(samples2, tolerance),
      "vmapSample should produce different results with different base keys"
    )
  }

  test("vmapSample Monte Carlo estimation accuracy") {
    val baseKey = Random.Key(111)
    val numSamples = 100

    // Test simple Monte Carlo estimation - compute mean of uniform samples
    val uniformSamples = Random.vmapSample(
      baseKey,
      numSamples,
      (key: Random.Key) => Random.uniform(Shape0, key) // Uniform in [0, 1)
    )

    val mean = uniformSamples.mean.toFloat
    val expectedMean = 0.5f // Expected mean of uniform [0, 1)

    // With 100 samples, should be reasonably close to 0.5
    val tolerance = 0.2f
    assertApproxEqual(mean, expectedMean, tolerance)

    println(s"Monte Carlo mean estimate with $numSamples samples: $mean (expected: $expectedMean)")
  }

  test("vmapSample with normal distribution sampling") {
    val baseKey = Random.Key(222)
    val numSamples = 100

    // Generate normal samples using Box-Muller transform
    val normalSamples = Random.vmapSample(baseKey, numSamples, (key: Random.Key) => Random.normal(Shape0, key))

    // Check basic properties of normal samples
    val mean = normalSamples.mean.toFloat
    val variance = normalSamples.variance.toFloat
    val stddev = math.sqrt(variance).toFloat

    // Standard normal should have mean ≈ 0 and std ≈ 1
    assertApproxEqual(mean, 0.0f, 0.3f) // Allow larger tolerance due to finite samples
    assertApproxEqual(stddev, 1.0f, 0.3f)

    println(s"Normal samples - mean: $mean, std: $stddev (expected: 0.0, 1.0)")
  }

  test("vmapSample performance vs sequential sampling") {
    val baseKey = Random.Key(333)
    val numSamples = 10

    // Time vmapSample approach
    val startVmap = System.nanoTime()
    val vmapResults = Random.vmapSample(
      baseKey,
      numSamples,
      (key: Random.Key) => Random.uniform(Shape1[Feature](100), key) // Larger tensors for meaningful timing
    )
    val vmapTime = System.nanoTime() - startVmap

    // Time sequential approach
    val startSeq = System.nanoTime()
    val keys = baseKey.split(numSamples)
    val seqResults = keys.map(key => Random.uniform(Shape1[Feature](100), key))
    val seqTime = System.nanoTime() - startSeq

    // Verify results are equivalent (same keys should produce same results)
    val stackedSeq = Tensor.stack[NewAxis = Feature](seqResults.head, seqResults.tail)
    assert(
      vmapResults.approxEquals(stackedSeq, tolerance),
      "vmapSample and sequential sampling should produce equivalent results"
    )

    println(s"vmapSample time: ${vmapTime / 1e6}ms, Sequential time: ${seqTime / 1e6}ms")

    // Note: We don't assert performance here as it depends on system and JAX compilation
    // But we log the times for manual verification
  }

  test("vmapSample with complex function") {
    val baseKey = Random.Key(444)
    val numSamples = 5

    // Test with a more complex sampling function
    val complexSamples = Random.vmapSample(
      baseKey,
      numSamples,
      (key: Random.Key) =>
        val keys = key.split(3)
        val x1 = Random.uniform(Shape0, keys(0))
        val x2 = Random.normal(Shape0, keys(1))
        val x3 = Random.uniform(Shape0, keys(2)) * Tensor0(10.0f)

        // Return a computed result
        x1 * x2 + x3.sin
    )

    // Just verify shape and that samples are different
    assert(complexSamples.shape.dims.length == 1)
    assert(complexSamples.shape.dims(0) == numSamples)

    // Check samples are different by comparing variance
    val variance = complexSamples.variance.toFloat

    assert(variance > tolerance, "Complex samples should have variance (be different)")
  }

end RandomTests
