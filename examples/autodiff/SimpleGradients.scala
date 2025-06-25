package examples.autodiff

import scala.language.experimental.namedTypeArguments
import shapeful.*

/**
 * Simple gradient computation examples.
 * 
 * This example demonstrates:
 * - Basic scalar gradients
 * - Vector gradients
 * - Gradients of composed functions
 * - Verification with finite differences
 */
object SimpleGradients extends App:

  println("=== Simple Automatic Differentiation Examples ===\n")

  type Feature = "feature"

  // 1. Simple scalar gradients
  println("1. Scalar Function Gradients")
  
  // f(x) = x^2
  val quadratic = (x: Tensor0) => x * x
  val quadraticGrad = Autodiff.grad(quadratic)
  
  val x = Tensor0(3.0f)
  val grad = quadraticGrad(x)
  
  println(s"f(x) = x^2")
  println(s"f($x) = ${quadratic(x)}")
  println(s"f'($x) = $grad (expected: ${2 * x.toFloat})")
  println()

  // 2. Cubic function
  println("2. Cubic Function")
  
  // f(x) = x^3 + 2x^2 + x + 1
  val cubic = (x: Tensor0) => x * x * x + Tensor0(2.0f) * x * x + x + Tensor0(1.0f)
  val cubicGrad = Autodiff.grad(cubic)
  
  val x2 = Tensor0(2.0f)
  val grad2 = cubicGrad(x2)
  val expected = 3 * x2.toFloat * x2.toFloat + 4 * x2.toFloat + 1  // 3x^2 + 4x + 1
  
  println(s"f(x) = x^3 + 2x^2 + x + 1")
  println(s"f($x2) = ${cubic(x2)}")
  println(s"f'($x2) = $grad2 (expected: $expected)")
  println()

  // 3. Exponential and trigonometric functions
  println("3. Transcendental Functions")
  
  // f(x) = sin(x) * exp(x)
  val transcendental = (x: Tensor0) => x.sin() * x.exp()
  val transcendentalGrad = Autodiff.grad(transcendental)
  
  val x3 = Tensor0(1.0f)
  val grad3 = transcendentalGrad(x3)
  
  println(s"f(x) = sin(x) * exp(x)")
  println(s"f($x3) = ${transcendental(x3)}")
  println(s"f'($x3) = $grad3")
  println(s"Note: f'(x) = cos(x)*exp(x) + sin(x)*exp(x) = exp(x)*(cos(x) + sin(x))")
  println()

  // 4. Vector function gradients
  println("4. Vector Function Gradients")
  
  // f(v) = ||v||^2 = sum(v^2)
  val norm2 = (v: Tensor1[Feature]) => (v * v).sum()
  val norm2Grad = Autodiff.grad(norm2)
  
  val vec = Tensor1[Feature](Seq(1.0f, 2.0f, 3.0f))
  val vecGrad = norm2Grad(vec)
  
  println(s"f(v) = ||v||^2 = sum(v^2)")
  println(s"v = $vec")
  println(s"f(v) = ${norm2(vec)}")
  println(s"∇f(v) = $vecGrad (expected: 2*v = ${(vec * Tensor0(2.0f))})")
  println()

  // 5. More complex vector function
  println("5. Complex Vector Function")
  
  // f(v) = sum(exp(v)) - max(v)^2
  val complexVec = (v: Tensor1[Feature]) => v.exp().sum() - v.max() * v.max()
  val complexVecGrad = Autodiff.grad(complexVec)
  
  val vec2 = Tensor1[Feature](Seq(0.0f, 1.0f, 2.0f))
  val complexGrad = complexVecGrad(vec2)
  
  println(s"f(v) = sum(exp(v)) - max(v)^2")
  println(s"v = $vec2")
  println(s"f(v) = ${complexVec(vec2)}")
  println(s"∇f(v) = $complexGrad")
  println()

  // 6. Finite difference verification
  println("6. Finite Difference Verification")
  
  val testFunc = (x: Tensor0) => x * x * x + x * x
  val analyticGrad = Autodiff.grad(testFunc)
  
  val testPoint = Tensor0(1.5f)
  val analytic = analyticGrad(testPoint)
  
  // Finite difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
  val h = 1e-5f
  val numerical = (testFunc(Tensor0(testPoint.toFloat + h)) - testFunc(Tensor0(testPoint.toFloat - h))) / Tensor0(2 * h)
  
  println(s"f(x) = x^3 + x^2")
  println(s"Analytic gradient at x=${testPoint.toFloat}: ${analytic.toFloat}")
  println(s"Numerical gradient at x=${testPoint.toFloat}: ${numerical.toFloat}")
  println(s"Difference: ${math.abs(analytic.toFloat - numerical.toFloat)}")
  println(s"Relative error: ${math.abs(analytic.toFloat - numerical.toFloat) / analytic.toFloat * 100}%")
  println()

  println("=== Simple Gradients Complete! ===")
