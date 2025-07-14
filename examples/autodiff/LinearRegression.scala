package examples.autodiff

import scala.language.experimental.namedTypeArguments
import shapeful.*
import shapeful.optimization.GradientDescent

/**
 * Simple linear regression example with vmap and automatic differentiation.
 * 
 * This example demonstrates:
 * - Real linear regression with input-output data
 * - Using vmap for vectorized operations
 * - Automatic differentiation for gradient computation
 * - Gradient descent optimization
 */
object LinearRegression extends App:

  println("=== Simple Linear Regression with vmap ===\n")

  type Feature = "feature"
  type Sample = "sample"

  // 1. Create synthetic dataset
  println("1. Creating Synthetic Dataset")
  
  val X = Tensor2[Sample, Feature](Seq(
    Seq(1.0f, 2.0f),
    Seq(2.0f, 3.0f), 
    Seq(3.0f, 4.0f),
    Seq(4.0f, 5.0f)
  ))
  val y = Tensor1[Sample](Seq(5.0f, 8.0f, 11.0f, 14.0f))  // y = x1 + 2*x2 + 1
  
  println(s"Training data X (${X.shape.dims}):")
  println(s"${X.inspect}")
  println(s"Training labels y: $y")
  println(s"True relationship: y = x1 + 2*x2 + 1")
  println()

  // 2. Initialize model parameters
  println("2. Initializing Model Parameters")
  
  var weight = Tensor1[Feature](Seq(0.1f, 0.1f))
  var bias = Tensor0(0.0f)
  
  println(s"Initial weight: $weight")
  println(s"Initial bias: ${bias.toFloat}")
  println()

  // 3. Define prediction function using vmap
  println("3. Defining Model with vmap")
  
  def predict(w: Tensor1[Feature], b: Tensor0, x: Tensor2[Sample, Feature]): Tensor1[Sample] =
    // Use vmap to apply dot product across all samples
    x.vmap[VmapAxis=Sample](sample => sample.dot(w) + b)

  // 4. Loss function (Mean Squared Error)
  def loss(w: Tensor1[Feature], b: Tensor0): Tensor0 =
    val predictions = predict(w, b, X)
    val errors = predictions - y
    (errors * errors).mean  // Mean squared error

  // 5. Get gradient function
  val gradFn = Autodiff.grad((params: (Tensor1[Feature], Tensor0)) =>
    val (w, b) = params
    loss(w, b)
  )
  
  val initialLoss = loss(weight, bias)
  println(s"Initial loss: ${initialLoss.toFloat}")
  println(s"Loss function: MSE = mean((predictions - labels)²)")
  println()

  // 6. Training loop
  println("4. Training with Gradient Descent")
  
  val learningRate = 0.01f
  val numEpochs = 150
  
  val optimizer = GradientDescent(learningRate)
  optimizer.optimize(gradFn, (weight, bias)).take(numEpochs)
  .zipWithIndex.foreach { case ((w, b), currentIteration) =>
    weight = w
    bias = b
    val currentLoss = loss(weight, bias)
    println(s"Epoch: $currentIteration, Loss: ${currentLoss.toFloat}, Weight: $weight, Bias: ${bias.toFloat}")
  }


  println()

  // 7. Final results
  println("5. Final Results")
  
  val finalLoss = loss(weight, bias)
  println(s"Final weight: $weight")
  println(s"Final bias: ${bias.toFloat}")
  println(s"Final loss: ${finalLoss.toFloat}")
  println()
  
  println("Expected parameters: weight ≈ [1.0, 2.0], bias ≈ 1.0")
  
  // Show final predictions vs true labels
  val finalPredictions = predict(weight, bias, X)
  println(s"Final predictions: $finalPredictions")
  println(s"True labels:      $y")
  println()
  
  println("Key features demonstrated:")
  println("- vmap for vectorized operations across the Sample dimension")
  println("- Automatic differentiation with Autodiff.grad")
  println("- Real linear regression with proper matrix operations")
  println("- Type-safe tensor dimensions with semantic labels")
  
  println("\n=== Linear Regression Complete! ===")
