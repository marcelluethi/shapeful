package examples.basic

import shapeful.*
import nn.*
import nn.ActivationFunctions.{relu, sigmoid}
import shapeful.Conversions.given
import shapeful.random.Random

object LogisticRegression:

  type Sample = "sample"
  type Feature = "feature"

  object BinaryLogisticRegression:
    case class Params(
      linearMap: LinearMap.Params[Feature]
    ) derives TensorTree, ToPyTree

  case class BinaryLogisticRegression(
    params: BinaryLogisticRegression.Params,
  ) extends Function[Tensor1[Feature], Tensor0]:
    private val linear = LinearMap(params.linearMap)
    def logits(input: Tensor1[Feature]): Tensor0 = linear(input)
    def probits(input: Tensor1[Feature]): Tensor0 = sigmoid(logits(input))
    def apply(input: Tensor1[Feature]): Tensor0 = logits(input) >= Tensor0(0f)

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
    val labelData = dfShuffled.column["species"].toArray.map(_.toFloat)

    val dataUnnormalized = Tensor2(Axis[Sample], Axis[Feature], featureData)
    val dataLabels = Tensor1(Axis[Sample], labelData)

    // TODO implement split
    val (trainingDataUnnormalized, valDataUnnormalized) = (dataUnnormalized, dataUnnormalized)
    val (trainLabels, valLabels) = (dataLabels, dataLabels)

    def calcMeanAndStd(t: Tensor2[Sample, Feature]): (Tensor1[Feature], Tensor1[Feature]) =
      val mean = t.vmap(Axis[Feature])(_.mean)
      val std = zipvmap(Axis[Feature])(t, mean):
        case (x, m) => 
          val epsilon = 1e-6f
          (x :- m).pow(2).mean.sqrt + epsilon
          // x.vmap(Axis[Sample])(xi => (xi - m).pow(2)).mean.sqrt + epsilon
      (mean, std)

    def standardizeData(mean: Tensor1[Feature], std: Tensor1[Feature])(data: Tensor2[Sample, Feature]): Tensor2[Sample, Feature] =
      data.vapply(Axis[Feature])(feature => (feature - mean) / std)
      // (data :- mean) :/ std

    val (trainMean, trainStd) = calcMeanAndStd(trainingDataUnnormalized)
    val trainingData = standardizeData(trainMean, trainStd)(trainingDataUnnormalized)
    val valData = standardizeData(trainMean, trainStd)(valDataUnnormalized)

    val (dataKey, trainKey) = Random.Key(42).split2()
    val (initKey, restKey) = trainKey.split2()
    val (lossKey, sampleKey) = restKey.split2()

    def loss(data: Tensor2[Sample, Feature])(params: BinaryLogisticRegression.Params): Tensor0 =
      val model = BinaryLogisticRegression(params)
      val losses = zipvmap(Axis[Sample])(data, trainLabels):
        case (sample, label) =>
          val logits = model.logits(sample)
          relu(logits) - logits * label + ((-logits.abs).exp + 1f).log
      losses.mean

    val initParams = BinaryLogisticRegression.Params(
      LinearMap.Params(initKey)(dataUnnormalized.dim(Axis[Feature]))
    )

    val trainLoss = loss(trainingData)
    val valLoss = loss(valData)
    val learningRate = 3e-1f
    val gd = GradientDescent(Autodiff.grad(trainLoss), learningRate)
    
    val trainTrajectory = Iterator.iterate(initParams)(gd.step)
    val finalParams = trainTrajectory
      .zipWithIndex
      .tapEach:
        case (params, index) =>
          val model = BinaryLogisticRegression(params)
          val trainPreds = trainingData.vmap(Axis[Sample])(model)
          val valPreds = valData.vmap(Axis[Sample])(model)
          println(List(
            "epoch: " + index,
            "trainAcc: " + (1 - (trainPreds - trainLabels).abs.mean),
            "valAcc: " + (1 - (valPreds - valLabels).abs.mean)
          ).mkString(", "))
      .map((params, _) => params)
      .drop(2500)
      .next()

    val finalModel = BinaryLogisticRegression(finalParams)
    val predictions = trainingData.vmap(Axis[Sample])(finalModel.probits)
    println(predictions)
    val predictionClasses = trainingData.vmap(Axis[Sample])(x => finalModel(x))

    println("\nTraining complete. Optimized parameters:" + finalParams)