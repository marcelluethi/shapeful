package examples

import shapeful.random.Random
import shapeful.*

// // Training function
// def train(params: VAEParams, batchData: Tensor2[Batch, Feature], key: Random.Key, numMCSamples: Int = 5): Tensor0 = {
//   val losses = batchData.vmap(Axis[Batch], sample => vae_loss(params, sample, key, numMCSamples))
//   losses.mean
// }
object DataUtils:
  def twoMoons[Sample <: Label, Feature <: Label, ClassLabel <: Label](
      numSamples: Int,
      key: Random.Key
  ): (Tensor2[Sample, Feature], Tensor1[ClassLabel]) =
    // Generate two moons dataset - two interleaving half circles
    val (dataKey, permutationKey) = key.split2()
    val (key1, key2) = dataKey.split2()
    val noiseKeys = key2.split(4)

    // Number of samples per moon
    val samplesPerMoon = numSamples / 2
    val remainingSamples = numSamples % 2

    // Generate first moon (upper crescent)
    val moon1Angles =
      Tensor.randUniform(
        key1,
        Shape1[Sample](samplesPerMoon),
        minval = Tensor0(0.0f),
        maxval = Tensor0(math.Pi.toFloat)
      )
    val moon1X = moon1Angles.cos
    val moon1Y = moon1Angles.sin

    // Generate second moon (lower crescent, shifted and rotated)
    val moon2Angles = Tensor.randUniform(
      key2,
      Shape1[Sample](samplesPerMoon + remainingSamples),
      minval = Tensor0(0.0f),
      maxval = Tensor0(math.Pi.toFloat)
    )
    val moon2X = moon2Angles.cos + Tensor0(1.0f) // Shift right by 1
    val moon2Y = moon2Angles.sin - Tensor0(0.5f) // Shift down by 0.5, then negate for lower crescent
    val moon2Y_flipped = moon2Y * Tensor0(-1.0f) // Flip vertically to create lower crescent

    // Add some noise for realism
    val noiseScale = 0.1f
    val noise1X = Tensor.randn(noiseKeys(0), Shape1[Sample](samplesPerMoon)) * Tensor0(noiseScale)
    val noise1Y = Tensor.randn(noiseKeys(1), Shape1[Sample](samplesPerMoon)) * Tensor0(noiseScale)
    val noise2X =
      Tensor.randn(noiseKeys(2), Shape1[Sample](samplesPerMoon + remainingSamples)) * Tensor0(noiseScale)
    val noise2Y =
      Tensor.randn(noiseKeys(3), Shape1[Sample](samplesPerMoon + remainingSamples)) * Tensor0(noiseScale)

    // Combine coordinates with noise using stack operation
    // Stack X and Y coordinates to create 2D points for each moon
    val moon1 = (moon1X + noise1X).stack(Axis[Feature], moon1Y + noise1Y).transpose
    val moon2 = (moon2X + noise2X).stack(Axis[Feature], moon2Y_flipped + noise2Y).transpose

    // Concatenate both moons along the sample dimension
    val X = moon1.concat(Axis[Sample], moon2)
    val y = Tensor
      .zeros(Shape1[ClassLabel](samplesPerMoon))
      .concat(Axis[ClassLabel], Tensor.ones(Shape1[ClassLabel](samplesPerMoon)))

    // Permute both X and y consistently along the Sample axis
    val permutedX = Random.permutation(Axis[Sample], permutationKey, X)
    val permutedY = Random.permutation(Axis[ClassLabel], permutationKey, y)

    (permutedX, permutedY)
