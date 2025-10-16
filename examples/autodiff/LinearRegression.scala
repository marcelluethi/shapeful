package examples.autodiff

import shapeful.*
import shapeful.autodiff.*
import shapeful.optimization.GradientDescent

/** Simple linear regression example with vmap and automatic differentiation.
  *
  * This example demonstrates:
  *   - Real linear regression with input-output data
  *   - Using vmap for vectorized operations
  *   - Automatic differentiation for gradient computation
  *   - Gradient descent optimization
  */
object LinearRegression extends App:

  println("=== Simple Linear Regression with vmap ===\n")

  type Feature = "feature"
  type Sample = "sample"

  // 1. Create synthetic dataset
  println("1. Creating Synthetic Dataset")

  val X = Tensor2[Sample, Feature](
    Seq(
      Seq(1.0f, 2.0f),
      Seq(2.0f, 3.0f),
      Seq(3.0f, 4.0f),
      Seq(4.0f, 5.0f)
    )
  )
  val y = Tensor1[Sample](Seq(5.0f, 8.0f, 11.0f, 14.0f)) // y = x1 + 2*x2 + 1

  println(s"Training data X (${X.shape.dims}):")
  println(s"${X.inspect}")
  println(s"Training labels y: $y")
  println(s"True relationship: y = x1 + 2*x2 + 1")
  println()

  // 2. Define model parameters
  println("2. Initializing Model Parameters")
  case class ModelParams(
      weight: Tensor1[Feature],
      bias: Tensor0
  ) derives TensorTree,
        ToPyTree

  // 3. Define prediction function using vmap
  println("3. Defining Model with vmap")

  def predict(params: ModelParams, x: Tensor2[Sample, Feature]): Tensor1[Sample] =
    // Use vmap to apply dot product across all samples
    x.vmap(Axis[Sample], sample => sample.dot(params.weight) + params.bias)

  // 4. Loss function (Mean Squared Error)
  def loss(params: ModelParams): Tensor0 =
    val predictions = predict(params, X)
    val errors = predictions - y
    (errors * errors).mean // Mean squared error

  // 5. Get gradient function
  val gradFn = Autodiff.grad(params => loss(params))

  val initialParams = ModelParams(
    weight = Tensor1[Feature](Seq(0.0f, 0.0f)), // Initial weights
    bias = Tensor0(0.0f) // Initial bias
  )

  val initialLoss = loss(initialParams)
  println(s"Initial loss: ${initialLoss.toFloat}")
  println(s"Loss function: MSE = mean((predictions - labels)²)")
  println()

  // 6. Training loop
  println("4. Training with Gradient Descent")

  val learningRate = 0.01f
  val numEpochs = 150

  val optimizer = GradientDescent(learningRate)
  val finalParams = optimizer
    .optimize(gradFn, initialParams)
    .take(numEpochs)
    .zipWithIndex
    .map { case (params, currentIteration) =>
      val currentLoss = loss(params)
      println(s"Epoch: $currentIteration, Loss: ${currentLoss.toFloat}, Params: $params")
      params
    }
    .toSeq
    .last

  println()

  // 7. Final results
  println("5. Final Results")

  val finalLoss = loss(finalParams)
  println(s"Final weight: ${finalParams.weight}")
  println(s"Final bias: ${finalParams.bias.toFloat}")
  println(s"Final loss: ${finalLoss.toFloat}")
  println()

  println("Expected parameters: weight ≈ [1.0, 2.0], bias ≈ 1.0")

  // Show final predictions vs true labels
  val finalPredictions = predict(finalParams, X)
  println(s"Final predictions: $finalPredictions")
  println(s"True labels:      $y")
  println()

  println("Key features demonstrated:")
  println("- vmap for vectorized operations across the Sample dimension")
  println("- Automatic differentiation with Autodiff.grad")
  println("- Real linear regression with proper matrix operations")
  println("- Type-safe tensor dimensions with semantic labels")

  println("\n=== Linear Regression Complete! ===")
