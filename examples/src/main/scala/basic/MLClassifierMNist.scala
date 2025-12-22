package examples.basic

import shapeful.*
import shapeful.Conversions.given
import nn.*
import nn.ActivationFunctions.{relu, sigmoid}
import shapeful.random.Random

import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.Try
import java.io.{FileInputStream, DataInputStream, BufferedInputStream}

def binaryCrossEntropy[L : Label](
  logits: Tensor1[L], label: Tensor0
): Tensor0 =
  val maxLogit = logits.max
  val stableExp = (logits :- maxLogit).exp
  val logSumExp = stableExp.sum.log + maxLogit
  val targetLogit = logits.slice(Axis[L] -> label.toInt)
  -(targetLogit - logSumExp)


object MLPClassifierMNist:

  trait Sample derives Label
  trait TrainSample extends Sample derives Label
  trait TestSample extends Sample derives Label
  trait Height derives Label
  trait Width derives Label
  trait Hidden derives Label
  trait Output derives Label

  object MLP:
    case class Params(
      layer1: LinearLayer.Params[Height |*| Width, Hidden],
      layer2: LinearLayer.Params[Hidden, Output],
    ) derives TensorTree, ToPyTree

    object Params:
      def apply(
        layer1Dim: Dim[Height |*| Width], layer2Dim: Dim[Hidden], outputDim: Dim[Output]
      )(
        paramKey: Random.Key
      ): Params = 
        val (key1, key2) = paramKey.split2()
        Params(
          layer1 = LinearLayer.Params(key1)(layer1Dim, layer2Dim),
          layer2 = LinearLayer.Params(key2)(layer2Dim, outputDim),
        )

  case class MLP(params: MLP.Params) extends Function[Tensor2[Height, Width], Tensor0]:
    
    private val layer1 = LinearLayer(params.layer1)
    private val layer2 = LinearLayer(params.layer2)

    def logits(
      image: Tensor2[Height, Width],
    ): Tensor1[Output] =
      val hidden = relu(layer1(image.ravel))
      layer2(hidden)

    override def apply(image: Tensor2[Height, Width]): Tensor0 = logits(image).argmax(Axis[Output])
    
  object MNISTLoader:

    private def readInt(dis: DataInputStream): Int = dis.readInt()
    private def loadImagePixels[S <: Sample : Label](filename: String, maxImages: Option[Int] = None): Try[Tensor3[S, Height, Width]] =
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
          val shape = Shape(Axis[S] -> numImages, Axis[Height] -> rows, Axis[Width] -> cols)
          val tensor = Tensor3(shape, allPixels, DType.Float32)
          tensor.toDevice(Device.CPU)
        finally dis.close()
      }

    private def loadLabelsArray[S <: Sample : Label](filename: String, maxLabels: Option[Int] = None): Try[Tensor1[S]] = Try {
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
        val tensor = Tensor1.fromInts(Axis[S], labels, DType.Int32)
        tensor.toDevice(Device.CPU)
      finally dis.close()
    }

    private def createDataset[S <: Sample : Label](imagesFile: String, labelsFile: String, maxSamples: Option[Int] = None): Try[Tuple2[Tensor[(S, Height, Width)], Tensor1[S]]] =
      for
        imagePixels <- loadImagePixels[S](imagesFile, maxSamples)
        labels <- loadLabelsArray[S](labelsFile, maxSamples)
      yield
        val numImages = imagePixels.shape(Axis[S])
        val numLabels = labels.shape.size
        if numImages != numLabels then
          throw new IllegalArgumentException(s"Mismatch: $numImages images vs $numLabels labels")
        println(s"Created in-memory MNIST dataset with $numImages images")
        (imagePixels, labels)

    def createTrainingDataset(dataDir: String = "data", maxSamples: Option[Int] = None): Try[Tuple2[Tensor[(TrainSample, Height, Width)], Tensor1[TrainSample]]] =
      val imagesFile = s"$dataDir/train-images-idx3-ubyte"
      val labelsFile = s"$dataDir/train-labels-idx1-ubyte"
      createDataset[TrainSample](imagesFile, labelsFile, maxSamples)

    def createTestDataset(dataDir: String = "data", maxSamples: Option[Int] = None): Try[Tuple2[Tensor[(TestSample, Height, Width)], Tensor1[TestSample]]] =
      val imagesFile = s"$dataDir/t10k-images-idx3-ubyte"
      val labelsFile = s"$dataDir/t10k-labels-idx1-ubyte"
      createDataset[TestSample](imagesFile, labelsFile, maxSamples)
    
  def main(args: Array[String]): Unit =

    val learningRate = 5e-2f
    val numSamples = 5120
    val batchSize = 512
    val numEpochs = 100
    val (dataKey, trainKey) = Random.Key(42).split2()
    val (initKey, restKey) = trainKey.split2()

    val (trainX, trainY) = MNISTLoader.createTrainingDataset(maxSamples = Some(numSamples)).get
    val (testX, testY) = MNISTLoader.createTestDataset(maxSamples = Some(1024)).get

    def batchLoss(batchImages: Tensor[(TrainSample, Height, Width)], batchLabels: Tensor1[TrainSample])(
        params: MLP.Params
    ): Tensor0 =
      val model = MLP(params)
      val losses = zipvmap(Axis[TrainSample])(batchImages, batchLabels):
        case (image, label) =>
          val logits = model.logits(image)
          binaryCrossEntropy(logits, label)
      losses.mean

    val initParams = MLP.Params(
      Axis[Height |*| Width] -> 28 * 28,
      Axis[Hidden] -> 128,
      Axis[Output] -> 10
    )(initKey)

    def accuracy[S <: Sample : Label](predictions: Tensor1[S], targets: Tensor1[S]): Tensor0 =
      val matches = zipvmap(Axis[S])(predictions, targets):
        case (pred, target) => Tensor0(pred.toInt == target.toInt)
      matches.mean

    def miniBatchGradientDescent(
      imageBatches: Seq[Tensor[(TrainSample, Height, Width)]],
      labelBatches: Seq[Tensor1[TrainSample]],
    )(
      params: MLP.Params
    ): MLP.Params =
      imageBatches.zip(labelBatches).foldLeft(params):
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
      trainY.chunk(Axis[TrainSample], batchSize),
    )
    val jitTrainMiniBatchGradientDescent = jit2(trainMiniBatchGradientDescent)
    val trainTrajectory = Iterator.iterate(initParams)( currentParams => 
      timed("Training"):
        trainMiniBatchGradientDescent(currentParams)
        // jitTrainMiniBatchGradientDescent(currentParams) // worse on CPU... TODO test on a GPU
    )
    val finalParams = trainTrajectory
      .zipWithIndex
      .tapEach:
        case (params, epoch) =>
          timed("Evaluation"):
            val model = MLP(params)
            val testPreds = testX.vmap(Axis[TestSample])(model)
            val testAccuracy = accuracy(testPreds, testY)
            val trainPreds = trainX.vmap(Axis[TrainSample])(model)
            val trainAccuracy = accuracy(trainPreds, trainY)
            println(List(
              s"Epoch $epoch",
              f"Test accuracy: ${testAccuracy.toFloat * 100}%.2f%%",
              f"Train accuracy: ${trainAccuracy.toFloat * 100}%.2f%%"
            ).mkString(", "))
      .map((params, _) => params)
      .drop(numEpochs)
      .next()

    println("\nTraining complete!")