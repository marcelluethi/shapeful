package examples.basic

import shapeful.*
import shapeful.autodiff.*
import shapeful.nn.*
import shapeful.jax.Jax
import shapeful.random.Random
import shapeful.optimization.GradientDescent
import examples.DataUtils
import io.github.quafadas.table.*
import scala.compiletime.constValueTuple
import scala.collection.immutable.ArraySeq

object LogisticRegression:

  type Sample = "sample"
  type Feature = "feature"
  type Output = "output"

  case class Params(
      weights: Tensor1[Feature],
      bias: Tensor0
  ) derives TensorTree,
        ToPyTree

  def initParams(key: Random.Key)(using featureDim: Dim[Feature]): Params =
    val keys = key.split(2)
    Params(
      weights = Tensor.zeros(Shape1[Feature](featureDim.dim)),
      bias = Tensor.zeros(Shape0)
    )

  def forward(params: Params, x: Tensor1[Feature]): (Tensor0, Tensor0) =
    val logits = x.dot(params.weights) + params.bias
    val probs = logits.sigmoid
    (logits, probs)

  def main(args: Array[String]): Unit =

    import io.github.quafadas.table.*
    val df = CSV
      .resource("penguins.csv", TypeInferrer.FromAllRows)
      .filter(row => !(row.species == 2))
      .toSeq

    val dfShuffled = scala.util.Random.shuffle(df)

    val learningRate = 3e-1f
    val numSamples = 1000
    val key = Random.Key(42)

    val numFeatures = 4
    given Dim[Feature] = Dim(numFeatures)

    val (dataKey, trainKey) = Random.Key(42).split2()
    val featureData = dfShuffled
      .map { row =>
        Array(
          row.flipper_length_mm.toFloat,
          row.bill_length_mm.toFloat,
          row.bill_depth_mm.toFloat,
          row.body_mass_g.toFloat
        )
      }
      .toArray
      .flatten
    val labelData = dfShuffled.column["species"].toArray.map(_.toFloat)

    val trainingDataUnnormalized = Tensor2.fromArray(
      Shape2[Sample, Feature](df.length, numFeatures),
      ArraySeq.unsafeWrapArray(featureData)
    )
    val labels = Tensor1.fromArray[Sample](ArraySeq.unsafeWrapArray(labelData))

    def standardize(t: Tensor2[Sample, Feature]): Tensor2[Sample, Feature] =
      val mean = t.vmap(Axis[Feature], _.mean)
      val std = t.zipVmap(Axis[Feature], mean)((x, m) => (x - m).pow(Tensor0(2f)).mean.sqrt + Tensor0(1e-6f))
      t.vmap(Axis[Sample], (x) => (x - mean) / std)
    val trainingData = standardize(trainingDataUnnormalized)

    val (initKey, restKey) = trainKey.split2()
    val (lossKey, sampleKey) = restKey.split2()

    def loss(p: Params): Tensor0 =
      val losses = trainingData.zipVmap(Axis[Sample], labels)((sample, label) =>
        val (logits, probs) = forward(p, sample)
        (logits.relu - logits * label + ((logits.abs * Tensor0(-1f)).exp + Tensor0(1f)).log)
      )
      losses.mean

    val initialParams = initParams(initKey)

    val gradFn = Autodiff.grad(loss)
    val finalParams = GradientDescent(learningRate)
      .optimize(gradFn, initialParams)
      .zipWithIndex
      .map((params, i) =>
        if i % 10 == 0 then
          println("loss: " + loss(params))
          val outputs = trainingData.vmap(Axis[Sample], x => forward(params, x)._2)
          println("acc: " + (Tensor0(1f) - (outputs - labels).abs.mean))
        end if
        params
      )
      .take(2500)
      .toSeq
      .last

    val predictions = trainingData.vmap(Axis[Sample], x => forward(finalParams, x)._2)
    println(predictions)
    val predictionClasses = predictions.vmap(Axis[Sample], p => p.argmax)

    println("\nTraining complete. Optimized parameters:" + finalParams)
