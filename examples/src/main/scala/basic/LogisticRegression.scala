package examples.basic

import shapeful.*
import nn.*
import nn.ActivationFunctions.{relu, sigmoid}
import shapeful.Conversions.given
import shapeful.random.Random

object LogisticRegression:

  trait Sample derives Label
  trait Feature derives Label

  object BinaryLogisticRegression:
    case class Params(
      linearMap: LinearMap.Params[Feature]
    ) derives TensorTree, ToPyTree

  case class BinaryLogisticRegression(
    params: BinaryLogisticRegression.Params,
  ) extends Function[Tensor1[Feature, Float32], Tensor0[Float32]]:
    private val linear = LinearMap(params.linearMap)
    def logits(input: Tensor1[Feature, Float32]): Tensor0[Float32] = linear(input)
    def probits(input: Tensor1[Feature, Float32]): Tensor0[Float32] = sigmoid(logits(input))
    def apply(input: Tensor1[Feature, Float32]): Tensor0[Float32] = logits(input) >= Tensor0.of[Float32](0f)
  def main(args: Array[String]): Unit =

    val df = PenguinCSV.read("./data/penguins.csv")
      .filter(row => row.species != 2)

    val dfShuffled = scala.util.Random.shuffle(df)

    val featureData = dfShuffled.map { row =>
        Array(
          row.flipperLengthMm.toFloat,
          row.billLengthMm.toFloat,
          row.billDepthMm.toFloat,
          row.bodyMassG.toFloat
        )
      }.toArray
    val labelData = dfShuffled.map(_.species.toFloat).toArray

    val dataUnnormalized = Tensor2.of[Float32](Axis[Sample], Axis[Feature], featureData)
    val dataLabels = Tensor1.of[Int32](Axis[Sample], labelData)

    // TODO implement split
    val (trainingDataUnnormalized, valDataUnnormalized) = (dataUnnormalized, dataUnnormalized)
    val (trainLabels, valLabels) = (dataLabels, dataLabels)

    def calcMeanAndStd(t: Tensor2[Sample, Feature, Float32]): (Tensor1[Feature, Float32], Tensor1[Feature, Float32]) =
      val mean = t.vmap(Axis[Feature])(_.mean)
      val std = zipvmap(Axis[Feature])(t, mean):
        case (x, m) => 
          val epsilon = 1e-6f
          (x :- m).pow(2f).mean.sqrt + epsilon
          // x.vmap(Axis[Sample])(xi => (xi - m).pow(2)).mean.sqrt + epsilon
      (mean, std)

    def standardizeData(mean: Tensor1[Feature, Float32], std: Tensor1[Feature, Float32])(data: Tensor2[Sample, Feature, Float32]): Tensor2[Sample, Feature, Float32] =
      data.vapply(Axis[Feature])(feature => (feature - mean) / std)
      // (data :- mean) :/ std

    val (trainMean, trainStd) = calcMeanAndStd(trainingDataUnnormalized)
    val trainingData = standardizeData(trainMean, trainStd)(trainingDataUnnormalized)
    val valData = standardizeData(trainMean, trainStd)(valDataUnnormalized)

    val (dataKey, trainKey) = Random.Key(42).split2()
    val (initKey, restKey) = trainKey.split2()
    val (lossKey, sampleKey) = restKey.split2()

    def loss(data: Tensor2[Sample, Feature, Float32])(params: BinaryLogisticRegression.Params): Tensor0[Float32] =
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
            "trainAcc: " + (1f - (trainPreds - trainLabels.asType[Float32]).abs.mean),
            "valAcc: " + (1f - (valPreds - valLabels.asType[Float32]).abs.mean)
          ).mkString(", "))
      .map((params, _) => params)
      .drop(2500)
      .next()

    val finalModel = BinaryLogisticRegression(finalParams)
    val predictions = trainingData.vmap(Axis[Sample])(finalModel.probits)
    println(predictions)
    val predictionClasses = trainingData.vmap(Axis[Sample])(x => finalModel(x))

    println("\nTraining complete. Optimized parameters:" + finalParams)

object PenguinCSV:
  import scala.io.Source
  import scala.util.Using

  case class PenguinRow(
    species: Int,
    billLengthMm: Double,
    billDepthMm: Double,
    flipperLengthMm: Double,
    bodyMassG: Double
  )

  def read(resourceName: String): Seq[PenguinRow] =
    Using.resource(Source.fromFile(resourceName)) { source =>
      val lines = source.getLines().toSeq
      val headers = lines.head.split(",").map(_.trim)
      val speciesIdx = headers.indexOf("species")
      val billLengthIdx = headers.indexOf("bill_length_mm")
      val billDepthIdx = headers.indexOf("bill_depth_mm")
      val flipperLengthIdx = headers.indexOf("flipper_length_mm")
      val bodyMassIdx = headers.indexOf("body_mass_g")
      
      lines.tail.flatMap { line =>
        val cols = line.split(",").map(_.trim)
        if (cols.length > bodyMassIdx) {
          try {
            Some(PenguinRow(
              species = cols(speciesIdx).toInt,
              billLengthMm = cols(billLengthIdx).toDouble,
              billDepthMm = cols(billDepthIdx).toDouble,
              flipperLengthMm = cols(flipperLengthIdx).toDouble,
              bodyMassG = cols(bodyMassIdx).toDouble
            ))
          } catch {
            case _: NumberFormatException => None
          }
        } else None
      }
    }