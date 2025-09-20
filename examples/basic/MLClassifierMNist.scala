package examples.basic

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.autodiff.*
import shapeful.nn.*
import shapeful.jax.Jax
import shapeful.random.Random
import shapeful.optimization.GradientDescent
import shapeful.nn.Layer.LayerDim
import examples.DataUtils
import examples.nn.losses.BinaryCrossEntropy
import examples.datautils.DataLoaderOps
import examples.datautils.MNISTLoader
import shapeful.tensor.TensorIndexing.*
import examples.datautils.DataLoader
import shapeful.nn.Utils
import scala.concurrent.ExecutionContext.Implicits.global

object MLPClassifierMNist:

  type Sample = "sample"

  type Feature = "feature"
  given LayerDim[Feature] = LayerDim(28 * 28)

  type Hidden1 = "hidden1"
  given LayerDim[Hidden1] = LayerDim(256)

  type Output = "output"
  given LayerDim[Output] = LayerDim(10)

  case class MLPParams(
      layer1: Linear.Params[Feature, Hidden1],
      output: Linear.Params[Hidden1, Output]
  ) derives TensorTree,
        ToPyTree

  def initParams(key: Random.Key): MLPParams =
    val keys = key.split(2)
    MLPParams(
      layer1 = Linear.he[Feature, Hidden1](keys(0)),
      output = Linear.xavier[Hidden1, Output](keys(1))
    )

  def forward(params: MLPParams, x: Tensor1[Feature]): (Tensor1[Output], Tensor1[Output]) =
    val layer1 = Linear[Feature, Hidden1]()
    val outputLayer = Linear[Hidden1, Output]()

    val mapping = layer1(params.layer1)
      .andThen(Activation.relu)
      .andThen(outputLayer(params.output))

    val logits = mapping(x)
    val probs = Activation.softmax[SoftmaxAxis = Output](logits)
    (logits, probs)

  def main(args: Array[String]): Unit =
    println("Starting optimized MNIST MLP training...")

    val learningRate = 5e-2f
    val numSamples = 20000
    val batchSize = 512
    val numEpochs = 100
    val key = Random.Key(42)
    val (dataKey, shuffleKey) = key.split2()

    // Load MNIST dataset using optimized memory-mapped loader
    val dataset = MNISTLoader.createTrainingDataset()
      .get
      .mapInput(t => t.reshape(Shape1[Feature](28 * 28)))
    println(s"Loaded memory-mapped dataset with ${dataset.size} images")

    // Split dataset indices for train/test (90/10 split)
    val (trainingData, testData) = DataLoaderOps.split(dataset, 0.9, shuffleKey) 

    // Loss function that works on a batch
    def batchLoss(batchImages: Tensor2[Sample, Feature], batchLabels: Tensor2[Sample, Output])(params: MLPParams): Tensor0 =
      val losses = batchImages.zipVmap[VmapAxis = Sample](batchLabels)((sample, label) =>
        val (logits, _) = forward(params, sample)
        BinaryCrossEntropy(logits, label)
      )
      shapeful.mean(losses)

    val initialParams = initParams(dataKey)
    var currentParams = initialParams

    // Training loop with epochs and batches
    for epoch <- 0 `until` numEpochs do
      println(s"Epoch $epoch")

      trainingData.batches(batchSize).foreach { (batchImages, batchLabels) =>
        if batchImages.size == batchSize then // Skip incomplete batches
          println("Processing batch...")
          val oneHotLabels = Utils.oneHot[Sample, Output](Tensor1.fromInts[Sample](batchLabels), 10)
          val batchImagesStacked = Tensor.stack[NewAxis = Sample](batchImages.head, batchImages.tail)
          val gradFn = Autodiff.grad(batchLoss(batchImagesStacked, oneHotLabels))
          currentParams = GradientDescent(learningRate).step(gradFn, currentParams)
        else
          println(s"Skipping incomplete batch of size ${batchImages.size}")
      }

    println("\nTraining complete!")

  private def calculateAccuracy(predictions: Tensor2[Sample, Output], targets: Tensor2[Sample, Output]): Float =
    // Convert predictions and targets to class indices and compare
    val predClasses = predictions.vmap[VmapAxis = Sample](pred => shapeful.argmax(pred))
    val targetClasses = targets.vmap[VmapAxis = Sample](target => shapeful.argmax(target))
    val matches = predClasses.zipVmap[VmapAxis = Sample](targetClasses)((pred, target) =>
      // This creates 1.0 if equal, 0.0 if different
      Tensor0(1.0f) - (pred - target).abs.sign
    )

    // Sum correct predictions and calculate accuracy
    val correct = shapeful.sum(matches)
    val total = predictions.shape.dim[Sample].toFloat
    correct.toFloat / total
