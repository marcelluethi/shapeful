package examples.autodiff

import scala.language.experimental.namedTypeArguments
import shapeful.*

/**
 * Complete linear regression example with gradient descent.
 * 
 * This example demonstrates:
 * - Defining a parameterized model
 * - Computing gradients for multiple parameters
 * - Implementing gradient descent optimization
 * - Training on synthetic data
 */
object LinearRegression extends App:

  println("=== Linear Regression with Automatic Differentiation ===\n")

  type Feature = "feature"
  type Sample = "sample"

  // Define our linear model parameters as a simple tuple
  type LinearModel = (Tensor1[Feature], Tensor0)  // (weight, bias)

  // 1. Generate synthetic training data
  println("1. Generating Synthetic Data")
  
  // True parameters: y = 2*x1 + 3*x2 - 1*x3 + 5
  val trueWeight = Tensor1[Feature](Seq(2.0f, 3.0f, -1.0f))
  val trueBias = Tensor0(5.0f)
  
  // For simplicity, we'll work with a very basic example
  println(s"True weight: $trueWeight")
  println(s"True bias: $trueBias")
  println()

  // 2. Define loss function
  println("2. Defining Loss Function")
  
  def simpleLoss(model: LinearModel): Tensor0 =
    val (weight, bias) = model
    // Simple quadratic loss for demonstration: (sum(weight) + bias - target)^2
    val target = Tensor0(10.0f)
    val prediction = weight.sum() + bias
    val diff = prediction - target
    diff * diff
  
  // Get gradient function
  val gradFunction = Autodiff.grad(simpleLoss)
  
  println("Loss function defined: (sum(weight) + bias - 10)^2")
  println()

  // 3. Initialize model parameters
  println("3. Initializing Model")
  
  var model: LinearModel = (
    Tensor1[Feature](Seq(0.1f, 0.1f, 0.1f)),  // Small random initialization
    Tensor0(0.0f)
  )
  
  println(s"Initial model:")
  println(s"  weight: ${model._1}")
  println(s"  bias: ${model._2}")
  
  val initialLoss = simpleLoss(model)
  println(s"  initial loss: ${initialLoss.toFloat}")
  println()

  // 4. Training loop
  println("4. Training with Gradient Descent")
  
  val learningRate = 0.01f
  val numEpochs = 10
  
  for epoch <- 1 to numEpochs do
    // Compute gradients
    val gradients = gradFunction(model)
    
    // Update parameters: param = param - learning_rate * gradient
    model = (
      model._1 - gradients._1 * Tensor0(learningRate),
      model._2 - gradients._2 * Tensor0(learningRate)
    )
    
    // Compute current loss
    val currentLoss = simpleLoss(model)
    
    if epoch % 2 == 0 || epoch == 1 then
      println(f"Epoch $epoch%2d: loss = ${currentLoss.toFloat}%.6f, weight = ${model._1}, bias = ${model._2.toFloat}%.4f")
  
  println()

  // 5. Final results
  println("5. Final Results")
  
  val finalLoss = simpleLoss(model)
  println(s"Final model:")
  println(s"  weight: ${model._1}")
  println(s"  bias: ${model._2}")
  println(s"  final loss: ${finalLoss.toFloat}")
  println()
  
  println("Note: This is a simplified example for demonstration.")
  println("The model should converge to weights that sum to ~10 to minimize loss.")
  println("A complete implementation would include:")
  println("- Proper matrix multiplication for predictions")
  println("- Real training data with input-output pairs")
  println("- Learning rate scheduling")
  println("- Validation and test sets")
  println("- More sophisticated optimization (Adam, etc.)")
  
  println("\n=== Linear Regression Complete! ===")
