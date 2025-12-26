package examples.basic

import shapeful.*
import nn.*
import nn.ActivationFunctions.{relu, sigmoid}
import shapeful.random.Random

object LogisticRegression:

  type Sample = "sample"
  type Feature = "feature"

  object BinaryLogisticRegression:
    case class Params(
        linearMap: LinearMap.Params[Feature]
    ) derives TensorTree, ToPyTree

  case class BinaryLogisticRegression(
      params: BinaryLogisticRegression.Params
  ) extends (FloatTensor1[Feature] => IntTensor0):
    private val linear = LinearMap[Feature, Float](params.linearMap)

    def logits(input: FloatTensor1[Feature]): FloatTensor0 =
      linear(input)
    def probits(input: FloatTensor1[Feature]): FloatTensor0 =
      sigmoid(logits(input))
    def apply(input: FloatTensor1[Feature]): IntTensor0 =
      (logits(input) >= Tensor0(0f)).asValue(TensorValue[Int])

  def main(args: Array[String]): Unit =

    import io.github.quafadas.table.*
    val df = CSV
      .resource("penguins.csv", TypeInferrer.FromAllRows)
      .filter(row => !(row.species == 2))
      .toSeq

    val dfShuffled = scala.util.Random.shuffle(df)

    val featureData = dfShuffled.map { row =>
      Array(
        row.flipper_length_mm.toFloat,
        row.bill_length_mm.toFloat,
        row.bill_depth_mm.toFloat,
        row.body_mass_g.toFloat
      )
    }.toArray
    val labelData = dfShuffled.column["species"].toArray

    val dataUnnormalized = FloatTensor2.fromArray(Axis[Sample], Axis[Feature], featureData)
    val dataLabels = IntTensor1.fromArray(Axis[Sample], labelData)

    // TODO implement split
    val (trainingDataUnnormalized, valDataUnnormalized) = (dataUnnormalized, dataUnnormalized)
    val (trainLabels, valLabels) = (dataLabels, dataLabels)

    def calcMeanAndStd(t: FloatTensor2[Sample, Feature]): (FloatTensor1[Feature], FloatTensor1[Feature]) =
      val mean = t.vmap(Axis[Feature])(_.mean)
      val epsilon = Tensor0[Float](1e-6f)
      val std = t.vmap(Axis[Feature]): feature =>
        val mean = feature.mean
        (feature :- mean).pow(Tensor0[Float](2f)).mean.sqrt + epsilon
      (mean, std)

    def standardizeData(mean: FloatTensor1[Feature], std: FloatTensor1[Feature])(
        data: FloatTensor2[Sample, Feature]
    ): FloatTensor2[Sample, Feature] =
      data.vapply(Axis[Feature])(feature => (feature - mean) / std)
      // (data :- mean) :/ std

    val (trainMean, trainStd) = calcMeanAndStd(trainingDataUnnormalized)
    val trainingData = standardizeData(trainMean, trainStd)(trainingDataUnnormalized)
    val valData = standardizeData(trainMean, trainStd)(valDataUnnormalized)

    val (dataKey, trainKey) = Random.Key(42).split2()
    val (initKey, restKey) = trainKey.split2()
    val (lossKey, sampleKey) = restKey.split2()

    def loss(data: FloatTensor2[Sample, Feature], labels: IntTensor1[Sample])(
        params: BinaryLogisticRegression.Params
    ): FloatTensor0 =
      val model = BinaryLogisticRegression(params)
      // Compute logits for all samples
      val allLogits = data.vmap(Axis[Sample])(model.logits)
      // Compute losses using vectorized operations
      val losses = zipvmap(Axis[Sample])(data, labels)((sample, label) =>
        val labelFloat = label.asValue(TensorValue[Float])
        val logits = model.logits(sample)
        relu(logits) - logits * labelFloat + ((-logits.abs).exp + FloatTensor0(1f)).log
      )
      losses.mean

    val initParams = BinaryLogisticRegression.Params(
      LinearMap.Params(initKey)(dataUnnormalized.dim(Axis[Feature]))
    )

    val trainLoss = loss(trainingData, trainLabels)
    val valLoss = loss(valData, valLabels)
    val learningRate = 3e-1f
    val gd = GradientDescent(Autodiff.grad(trainLoss), learningRate)

    val trainTrajectory = Iterator.iterate(initParams)(gd.step)
    val finalParams = trainTrajectory.zipWithIndex
      .tapEach:
        case (params, index) =>
          val model = BinaryLogisticRegression(params)
          val trainPreds = trainingData.vmap(Axis[Sample])(model)
          val valPreds = valData.vmap(Axis[Sample])(model)
          println(
            List(
              "epoch: " + index,
              "loss: " + loss(trainingData, trainLabels)(params).item,
              "trainAcc: " + (FloatTensor0(1f) - (trainPreds - trainLabels).abs.mean).item,
              "valAcc: " + (FloatTensor0(1f) - (valPreds - valLabels).abs.mean).item
            ).mkString(", ")
          )
      .map((params, _) => params)
      .drop(2500)
      .next()

    val finalModel = BinaryLogisticRegression(finalParams)
    val predictions = trainingData.vmap(Axis[Sample])(finalModel.probits)
    println(predictions)
    val predictionClasses = trainingData.vmap(Axis[Sample])(x => finalModel(x))

    println("\nTraining complete. Optimized parameters:" + finalParams)
