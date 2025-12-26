package examples.basic

import shapeful.*
import nn.*
import nn.ActivationFunctions.{relu, sigmoid}
import shapeful.random.Random

import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.Try
import java.io.{FileInputStream, DataInputStream, BufferedInputStream}

def binaryCrossEntropy[L: Label](
    logits: FloatTensor1[L],
    label: Tensor0[Int]
): Tensor0[Float] =
  val maxLogit = logits.max
  val stableExp = (logits :- maxLogit).exp
  val logSumExp = stableExp.sum.log + maxLogit
  val targetLogit = logits.slice(Axis[L] -> label.item)
  -(targetLogit - logSumExp)

object MLPClassifierMNist:

  type TrainSample = "train-sample"
  type TestSample = "test-sample"
  type Height = "height"
  type Width = "width"
  type Hidden = "hidden"
  type Output = "output"

  object MLP:
    case class Params(
        layer1: LinearLayer.Params[Height * Width, Hidden, Float],
        layer2: LinearLayer.Params[Hidden, Output, Float]
    ) derives TensorTree,
          ToPyTree

    object Params:
      def apply(
          layer1Dim: Dim[Height * Width],
          layer2Dim: Dim[Hidden],
          outputDim: Dim[Output]
      )(
          paramKey: Random.Key
      ): Params =
        val (key1, key2) = paramKey.split2()
        Params(
          layer1 = LinearLayer.Params(key1)(layer1Dim, layer2Dim),
          layer2 = LinearLayer.Params(key2)(layer2Dim, outputDim)
        )

  case class MLP(params: MLP.Params) extends Function[FloatTensor2[Height, Width], IntTensor0]:

    private val layer1 = LinearLayer[Height * Width, Hidden, Float](params.layer1)
    private val layer2 = LinearLayer[Hidden, Output, Float](params.layer2)

    def logits(
        image: Tensor2[Height, Width, Float]
    ): Tensor1[Output, Float] =
      val hidden = relu(layer1(image.ravel))
      layer2(hidden)

    override def apply(image: FloatTensor2[Height, Width]): IntTensor0 = logits(image).argmax(Axis[Output])

  object MNISTLoader:

    private type Sample = "sample"

    private def readInt(dis: DataInputStream): Int = dis.readInt()
    private def loadImagePixels(
        filename: String,
        maxImages: Option[Int] = None
    ): Try[Tensor3[Sample, Height, Width, Float]] =
      Try {
        val dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filename)))
        try
          val magic = readInt(dis)
          if magic != 2051 then
            throw new IllegalArgumentException(s"Invalid magic number for images: $magic (expected 2051)")

          val totalImages = readInt(dis)
          val rows = readInt(dis)
          val cols = readInt(dis)

          val numImages = maxImages.map(max => math.min(max, totalImages)).getOrElse(totalImages)
          println(s"Loading $numImages of $totalImages images (${rows}x${cols}) from $filename into memory as Tensor3")

          // Read all pixel data at once
          val totalPixels = numImages * rows * cols
          val pixelBytes = new Array[Byte](totalPixels)
          dis.readFully(pixelBytes)
          // Convert bytes to floats with vectorized operation
          val allPixels = pixelBytes.map(b => (b & 0xff) / 255.0f)
          val shape = Shape(Axis[Sample] -> numImages, Axis[Height] -> rows, Axis[Width] -> cols)
          val tensor = FloatTensor.fromArray(shape, allPixels)
          tensor.toDevice(Device.CPU)
        finally dis.close()
      }

    private def loadLabelsArray(filename: String, maxLabels: Option[Int] = None): Try[Tensor1[Sample, Int]] = Try {
      val dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filename)))
      try
        val magic = readInt(dis)
        if magic != 2049 then
          throw new IllegalArgumentException(s"Invalid magic number for labels: $magic (expected 2049)")

        val totalLabels = readInt(dis)
        val numLabels = maxLabels.map(max => math.min(max, totalLabels)).getOrElse(totalLabels)
        println(s"Loading $numLabels of $totalLabels labels from $filename into memory as Tensor1")

        val labels = Array.ofDim[Int](numLabels)
        for i <- 0.until(numLabels) do labels(i) = dis.readUnsignedByte()

        // Create Tensor1 from labels - specify the label type correctly
        val tensor = IntTensor1.fromArray(Axis[Sample], labels)
        tensor.toDevice(Device.CPU)
      finally dis.close()
    }

    private def createDataset(
        imagesFile: String,
        labelsFile: String,
        maxSamples: Option[Int] = None
    ): Try[Tuple2[Tensor[(Sample, Height, Width), Float], Tensor1[Sample, Int]]] =
      for
        imagePixels <- loadImagePixels(imagesFile, maxSamples)
        labels <- loadLabelsArray(labelsFile, maxSamples)
      yield
        val numImages = imagePixels.shape(Axis[Sample])
        val numLabels = labels.shape.size
        if numImages != numLabels then
          throw new IllegalArgumentException(s"Mismatch: $numImages images vs $numLabels labels")
        println(s"Created in-memory MNIST dataset with $numImages images")
        (imagePixels, labels)

    def createTrainingDataset(
        dataDir: String = "data",
        maxSamples: Option[Int] = None
    ): Try[Tuple2[Tensor[(TrainSample, Height, Width), Float], Tensor1[TrainSample, Int]]] =
      val imagesFile = s"$dataDir/train-images-idx3-ubyte"
      val labelsFile = s"$dataDir/train-labels-idx1-ubyte"
      val dataset = createDataset(imagesFile, labelsFile, maxSamples)
      dataset.map:
        case (images, labels) =>
          (images.relabel(Axis[Sample] -> Axis[TrainSample]), labels.relabel(Axis[Sample] -> Axis[TrainSample]))

    def createTestDataset(
        dataDir: String = "data",
        maxSamples: Option[Int] = None
    ): Try[Tuple2[Tensor[(TestSample, Height, Width), Float], Tensor1[TestSample, Int]]] =
      val imagesFile = s"$dataDir/t10k-images-idx3-ubyte"
      val labelsFile = s"$dataDir/t10k-labels-idx1-ubyte"
      val dataset = createDataset(imagesFile, labelsFile, maxSamples)
      dataset.map:
        case (images, labels) =>
          (images.relabel(Axis[Sample] -> Axis[TestSample]), labels.relabel(Axis[Sample] -> Axis[TestSample]))

  def main(args: Array[String]): Unit =

    val learningRate = 5e-2f
    val numSamples = 5120
    val batchSize = 512
    val numEpochs = 100
    val (dataKey, trainKey) = Random.Key(42).split2()
    val (initKey, restKey) = trainKey.split2()

    val (trainX, trainY) = MNISTLoader.createTrainingDataset(maxSamples = Some(numSamples)).get
    val (testX, testY) = MNISTLoader.createTestDataset(maxSamples = Some(1024)).get

    def batchLoss(batchImages: FloatTensor[(TrainSample, Height, Width)], batchLabels: IntTensor1[TrainSample])(
        params: MLP.Params
    ): Tensor0[Float] =
      val model = MLP(params)
      val losses = zipvmap(Axis[TrainSample])(batchImages, batchLabels): (image, label) =>
        val logits = model.logits(image)
        binaryCrossEntropy(logits, label)
      losses.mean

    val initParams = MLP.Params(
      Axis[Height * Width] -> 28 * 28,
      Axis[Hidden] -> 128,
      Axis[Output] -> 10
    )(initKey)

    def accuracy[Sample: Label](
        predictions: IntTensor1[Sample],
        targets: IntTensor1[Sample]
    ): Tensor0[Float] =
      val matches = zipvmap(Axis[Sample])(predictions, targets)(_ === _)
      matches.mean

    def miniBatchGradientDescent(
        imageBatches: Seq[Tensor[(TrainSample, Height, Width), Float]],
        labelBatches: Seq[Tensor1[TrainSample, Int]]
    )(
        params: MLP.Params
    ): MLP.Params =
      imageBatches
        .zip(labelBatches)
        .foldLeft(params):
          case (params, (imageBatch, labelBatch)) =>
            val lossBatch = batchLoss(imageBatch, labelBatch)
            val df = Autodiff.grad(lossBatch)
            GradientDescent(df, learningRate).step(params)

    def timed[A](template: String)(block: => A): A =
      val t0 = System.currentTimeMillis()
      val result = block
      println(s"$template took ${System.currentTimeMillis() - t0} ms")
      result

    val trainMiniBatchGradientDescent = miniBatchGradientDescent(
      trainX.chunk(Axis[TrainSample], batchSize),
      trainY.chunk(Axis[TrainSample], batchSize)
    )
    val jitTrainMiniBatchGradientDescent = jit2(trainMiniBatchGradientDescent)
    val trainTrajectory = Iterator.iterate(initParams)(currentParams =>
      timed("Training"):
        trainMiniBatchGradientDescent(currentParams)
        // jitTrainMiniBatchGradientDescent(currentParams) // worse on CPU... TODO test on a GPU
    )
    val finalParams = trainTrajectory.zipWithIndex
      .tapEach:
        case (params, epoch) =>
          timed("Evaluation"):
            val model = MLP(params)
            val testPreds = testX.vmap(Axis[TestSample])(model)
            val testAccuracy = accuracy(testPreds, testY).item
            val trainPreds = trainX.vmap(Axis[TrainSample])(model)
            val trainAccuracy = accuracy(trainPreds, trainY).item
            println(
              List(
                s"Epoch $epoch",
                f"Test accuracy: ${testAccuracy * 100}%.2f%%",
                f"Train accuracy: ${trainAccuracy * 100}%.2f%%"
              ).mkString(", ")
            )
      .map((params, _) => params)
      .drop(numEpochs)
      .next()

    println("\nTraining complete!")
