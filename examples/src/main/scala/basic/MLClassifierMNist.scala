package examples.basic

import shapeful.*
import shapeful.autodiff.*
import shapeful.nn.*
import shapeful.jax.Jax
import shapeful.random.Random
import shapeful.optimization.GradientDescent
import examples.DataUtils
import examples.nn.losses.BinaryCrossEntropy
import examples.datautils.DataLoaderOps
import examples.datautils.MNISTLoader
import shapeful.tensor.TensorIndexing.*
import examples.datautils.DataLoader
import shapeful.nn.Utils
import scala.concurrent.ExecutionContext.Implicits.global
import shapeful.tensor.Device
import shapeful.jax.Jit

object MLPClassifierMNist:

  type Sample = "sample"

  type Feature = "feature"
  given Dim[Feature] = Dim(28 * 28)

  type Hidden1 = "hidden1"
  given Dim[Hidden1] = Dim(256)

  type Output = "output"
  given Dim[Output] = Dim(10)

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
    val probs = Activation.softmax(Axis[Output], logits)
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
    val dataset = MNISTLoader
      .createTrainingDataset()
      .get
    println(s"Loaded memory-mapped dataset with ${dataset.size} images")

    // Split dataset indices for train/test (90/10 split)
    val (trainingData, testData) = DataLoaderOps.split(dataset, 0.9)

    // Loss function that works on a batch
    def batchLoss(batchImages: Tensor2[Sample, Feature], batchLabels: Tensor2[Sample, Output])(
        params: MLPParams
    ): Tensor0 =
      val losses = batchImages.zipVmap(Axis[Sample])(batchLabels) { (sample, label) =>
        val (logits, _) = forward(params, sample)
        BinaryCrossEntropy(logits, label)
      }
      shapeful.mean(losses)

    val initialParams = initParams(dataKey)
    var currentParams = initialParams

    type BatchOutput = (Sample, Output)

    // JIT-compile accuracy calculation
    def accuracyFn(predictions: Tensor2[Sample, Output], targets: Tensor2[Sample, Output]): Tensor0 =
      val predClasses = predictions.vmap(Axis[Sample]) { pred => shapeful.argmax(pred) }
      val targetClasses = targets.vmap(Axis[Sample]) { target => shapeful.argmax(target) }
      val matches =
        predClasses.zipVmap(Axis[Sample])(targetClasses) { (pred, target) => Tensor0(1.0f) - (pred - target).abs.sign }
      shapeful.sum(matches)

    val jittedAccuracy = Jit.function2(accuracyFn)

    val jittedGradStep = Jit.gradientStep(
      (params: MLPParams, flattenedImages: Tensor2[Sample, Feature], oneHotLabels: Tensor2[Sample, Output]) =>
        val gradFn = Autodiff.grad(batchLoss(flattenedImages, oneHotLabels))
        GradientDescent(learningRate).step(gradFn, params)
    )

    // Training loop with epochs and batches
    for epoch <- 0 `until` numEpochs do
      println(s"Epoch $epoch")

      trainingData.batches[Sample](batchSize).zipWithIndex.foreach { case ((batchImages, batchLabels), batchIndex) =>
        val actualBatchSize = batchImages.shape.dim[Sample]
        batchImages.toDevice(Device.GPU)
        batchLabels.toDevice(Device.GPU)
        if actualBatchSize == batchSize then
          if batchIndex % 30 == 0 then

            val (testImages, testLabels) =
              testData.getBatch[Sample](0, 1000) // Use first 1000 test samples for quick eval
            val testProbs = testImages.vmap(Axis[Sample]) { image =>
              val flattened = image.reshape(Shape(Axis[Feature] -> 28 * 28))
              forward(currentParams, flattened)._2
            }
            val oneHotTestLabels =
              Utils.oneHot[Sample, Output](testLabels.reshape(Shape(Axis[Sample] -> testLabels.shape.dim[Sample])), 10)

            val correctCount = jittedAccuracy(testProbs, oneHotTestLabels)
            val accuracy = correctCount.toFloat / 1000.0f

            println(f"Test accuracy after batch $batchIndex: ${accuracy * 100}%.2f%%")
            println("done evaluating")

          // Flatten images for MLP input
          val flattenedImages = batchImages.vmap(Axis[Sample]) { image => image.reshape(Shape(Axis[Feature] -> 28 * 28)) }
          val oneHotLabels =
            Utils.oneHot[Sample, Output](batchLabels.reshape(Shape(Axis[Sample] -> batchLabels.shape.dim[Sample])), 10)

          // Use JIT-compiled gradient step - FAST! 10-50x speedup!
          currentParams = jittedGradStep(currentParams, flattenedImages, oneHotLabels)
        else println(s"Skipping incomplete batch of size $actualBatchSize")
      }

    println("\nTraining complete!")
