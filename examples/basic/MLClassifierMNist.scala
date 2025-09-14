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

object MLPClassifierMNist {

  type Sample = "sample"

  type Feature = "feature"
  given LayerDim[Feature] = LayerDim(28 * 28)
 
  type Hidden1 = "hidden1"
  given LayerDim[Hidden1] = LayerDim(10)

  type Output = "output"
  given LayerDim[Output] = LayerDim(10)

    
  case class MLPParams(
    layer1: Linear.Params[Feature, Hidden1],
    output: Linear.Params[Hidden1, Output]
  ) derives TensorTree, ToPyTree
    
  def initParams(key: Random.Key): MLPParams = {
      val keys = key.split(2)
      MLPParams(
          layer1 = Linear.he[Feature, Hidden1](keys(0)),
          output = Linear.xavier[Hidden1, Output](keys(1))
      )
  }

  def forward(params: MLPParams, x: Tensor1[Feature]): (Tensor1[Output], Tensor1[Output]) = {
    val layer1 = Linear[Feature, Hidden1]()
    val outputLayer = Linear[Hidden1, Output]()
    
    val mapping = layer1(params.layer1)
    .andThen(Activation.relu)
    .andThen(outputLayer(params.output))

    val logits = mapping(x)
    val probs = Activation.softmax[SoftmaxAxis=Output](logits)
    (logits, probs)
  }
  
  def main(args: Array[String]): Unit = {
  val learningRate = 5e-2f
  val numSamples = 5000
  val batchSize = 32  // Add batch size
  val numEpochs = 100  // Add number of epochs
  val key = Random.Key(42)
  val (dataKey, shuffleKey) = key.split2()

  // Load MNIST dataset using DataLoader trait
  val dataset = MNISTLoader.createTrainingDataset(dataDir = "data", maxImages = numSamples).get
  
  // Use DataLoader operations for elegant data handling
  val (trainDataset, testDataset) = DataLoaderOps.split(dataset, trainRatio = 0.9, shuffleKey)
  
  // Convert test data to tensors (only once since we don't batch test data)
  val (testImages, testLabels) = DataLoaderOps.toSeqs(testDataset)
  val testImagesTensor = Tensor.stack[NewAxis = Sample](testImages.head, testImages.tail)
  val testLabelsOneHotSeq = testLabels.map(label => Utils.oneHot[Classes=Output](Tensor0(label.toFloat), 10))    
  val testLabelsOneHot = Tensor.stack[NewAxis = Sample](testLabelsOneHotSeq.head, testLabelsOneHotSeq.tail)
  val flattenedTestImages = testImagesTensor.vmap[VmapAxis = Sample](img => img.reshape(Shape1[Feature](28 * 28)))

  // Helper function to convert a batch to tensors
  def batchToTensors(batchImages: LazyList[Tensor2["height", "width"]], batchLabels: LazyList[Int]): (Tensor2[Sample, Feature], Tensor2[Sample, Output]) = {
    val imageSeq = batchImages.toSeq
    val labelSeq = batchLabels.toSeq
    
    val imagesTensor = Tensor.stack[NewAxis = Sample](imageSeq.head, imageSeq.tail)
    val labelsOneHotSeq = labelSeq.map(label => Utils.oneHot[Classes=Output](Tensor0(label.toFloat), 10))
    val labelsOneHot = Tensor.stack[NewAxis = Sample](labelsOneHotSeq.head, labelsOneHotSeq.tail)
    val flattenedImages = imagesTensor.vmap[VmapAxis = Sample](img => img.reshape(Shape1[Feature](28 * 28)))
    
    (flattenedImages, labelsOneHot)
  }

  // Loss function that works on a batch
  def batchLoss(batchImages: Tensor2[Sample, Feature], batchLabels: Tensor2[Sample, Output])(params: MLPParams): Tensor0 = {
    val losses = batchImages.zipVmap[VmapAxis = Sample](batchLabels)((sample, label) => 
      val (logits, _) = forward(params, sample)
      BinaryCrossEntropy(logits, label)
    )
    losses.mean
  }
  
  val initialParams = initParams(dataKey)
  
  var currentParams = initialParams
  
  // Training loop with epochs and batches
  for (epoch <- 0 until numEpochs) {
    println(s"Epoch $epoch")
    val loss = batchLoss(flattenedTestImages, testLabelsOneHot)(currentParams)
    println(s"Loss: $loss")
    
    // Calculate accuracy on test set
    val outputs = flattenedTestImages.vmap[VmapAxis = Sample](x => forward(currentParams, x)._2)
    val accuracy = calculateAccuracy(outputs, testLabelsOneHot)
    println(s"Accuracy: $accuracy")
    System.gc()
      
    // Shuffle dataset for each epoch
    val epochKey = Random.Key(42 + epoch) // Use a deterministic but different key for each epoch
    val shuffledDataset = trainDataset.shuffle(epochKey)
    
    // Iterate through batches
    for ((batchImages, batchLabels) <- shuffledDataset.batches(batchSize)) {
      val (imagesTensor, labelsTensor) = batchToTensors(batchImages, batchLabels)
      
      // Create gradient function for this batch
      val gradFn = Autodiff.grad(batchLoss(imagesTensor, labelsTensor))
      
      currentParams = GradientDescent(learningRate).step(gradFn, currentParams)
    
    }
  }

  println("\nTraining complete!")
}
  
  private def calculateAccuracy(predictions: Tensor2[Sample, Output], targets: Tensor2[Sample, Output]): Float = 
    // Convert predictions and targets to class indices and compare
    val predClasses = predictions.vmap[VmapAxis = Sample](_.argmax)
    val targetClasses = targets.vmap[VmapAxis = Sample](_.argmax)
    val matches = predClasses.zipVmap[VmapAxis = Sample](targetClasses)((pred, target) => 
        // This creates 1.0 if equal, 0.0 if different
        Tensor0(1.0f) - (pred - target).abs.sign
      )
      
      // Sum correct predictions and calculate accuracy
      val correct = matches.sum
      val total = predictions.shape.dim[Sample].toFloat
      correct.toFloat / total

}

