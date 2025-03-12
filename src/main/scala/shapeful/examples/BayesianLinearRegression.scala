package shapeful.examples


// Example usage:

import shapeful.distributions.NormalDistribution
import shapeful.tensor.{Dimension, Tensor, addScalar, cov, dot, mult, multScalar, pow, sub, add}
import shapeful.tensor.Dimension.Symbolic
import shapeful.tensor.Tensor.{Tensor0, Tensor1, Tensor2}
import shapeful.autodiff.Autodiff
import shapeful.autodiff.TensorTupleOps
import shapeful.optimization.GradientDescent


def bayesianLinearRegression(xs : Seq[Float], y : Seq[Float]) : Unit =

  type Data = "data"
  given Dimension[Data] = Symbolic[Data](xs.size)

  type Space = "space"
  given Dimension[Space] = Symbolic[Space](1)

  val X = Tensor.fromSeq[(Data, Space)](xs, requiresGrad = false)
  val yt = Tensor.fromSeq[Tuple1[Data]](y, requiresGrad = false)

  val pw = NormalDistribution[Tuple1[Space]](Tensor(0f, requiresGrad = false), Tensor(10f, requiresGrad = false))
  val pb = NormalDistribution[EmptyTuple](Tensor(0f, requiresGrad = false), Tensor(100f, requiresGrad = false))

  def likelihood(w : Tensor1[Space], b : Tensor0) : Tensor0 =
    val mu : Tensor1[Data] = X.dot(w).addScalar(b)
    val sigma : Tensor1[Data] = Tensor(1, requiresGrad = false)

    NormalDistribution[Tuple1[Data]](mu, sigma).logpdf(yt).mean[Data]

  def f(w : Tensor1[Space], b: Tensor0) : Tensor0 =
    val logpw = pw.logpdf(w).sum[Space]
    val logpb = pb.logpdf(b).sum[Space]

    val ll = likelihood(w, b)
    logpw.add(logpb).add(ll)



  // optimize
  val lr : Tensor0 = Tensor(0.01f, requiresGrad = false)
  val w0 : Tensor1[Space] = Tensor(0.0, requiresGrad = true)
  val b0 : Tensor0 = Tensor(0.0, requiresGrad = true)

  val diffFun = Autodiff.deriv(f)
  val gd = GradientDescent(0.0005)

  val (w, b) = gd.optimize(diffFun, (w0, b0)).zipWithIndex.map { case (z, i) =>
    if i % 100 == 0 then
        println(s"Iteration $i: $z")
    z
  }.drop(100000).take(1).toSeq.last

  println(s"w: $w b: $b")

@main def runBayesianLinearRegression() : Unit = 

    val w = 1f
    val b = 5f

    val xs = Seq.tabulate(100)(x => x.toFloat)
    val ys = xs.map(x => w * x + b)

    bayesianLinearRegression(xs, ys)
    println("done")