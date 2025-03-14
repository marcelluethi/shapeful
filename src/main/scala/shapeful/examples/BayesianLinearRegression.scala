package shapeful.examples


// Example usage:

import shapeful.distributions.Normal
import shapeful.tensor.Dimension.Symbolic
import shapeful.tensor.Tensor.{Tensor0, Tensor1, Tensor2}

import shapeful.tensor.TensorOps.* 
import shapeful.tensor.Tensor0Ops.* 
import shapeful.tensor.Tensor2Ops.* 

import shapeful.autodiff.Autodiff
import shapeful.tensor.TensorTupleOps
import shapeful.optimization.GradientOptimizer
import shapeful.inference.MetropolisHastings
import shapeful.tensor.Dimension
import shapeful.tensor.Tensor


def bayesianLinearRegression(xs : Seq[Float], y : Seq[Float]) : Unit =

  type Data = "data"
  given Dimension[Data] = Symbolic[Data](xs.size)

  type Space = "space"
  given Dimension[Space] = Symbolic[Space](1)

  val X = Tensor.fromSeq[(Data, Space)](xs, requiresGrad = false)
  val yt = Tensor.fromSeq[Tuple1[Data]](y, requiresGrad = false)

  val pw = Normal[Tuple1[Space]](Tensor(0f, requiresGrad = false), Tensor(10f, requiresGrad = false))
  val pb = Normal[EmptyTuple](Tensor(0f, requiresGrad = false), Tensor(100f, requiresGrad = false))

  def likelihood(w : Tensor1[Space], b : Tensor0) : Tensor0 =
    val mu : Tensor1[Data] = X.matmul(w).add(b)
    val sigma : Tensor1[Data] = Tensor(1, requiresGrad = false)

    Normal[Tuple1[Data]](mu, sigma).logpdf(yt).mean[Data]

  def f(w : Tensor1[Space], b: Tensor0) : Tensor0 =
    val logpw = pw.logpdf(w).sum[Space]
    val logpb = pb.logpdf(b).sum[Space]

    val ll = likelihood(w, b)
    logpw.add(logpb).add(ll)



  val w0 : Tensor1[Space] = Tensor(0.0, requiresGrad = true)
  val b0 : Tensor0 = Tensor(0.0, requiresGrad = true)

  // Finding map solution
  
  // val lr : Tensor0 = Tensor(0.01f, requiresGrad = false)
  // val diffFun = Autodiff.deriv(f)
  // val gd = GradientDescent(0.0005)

  // val (w, b) = gd.optimize(diffFun, (w0, b0)).zipWithIndex.map { case (z, i) =>
  //   if i % 100 == 0 then
  //       println(s"Iteration $i: $z")
  //   z
  // }.drop(100000).take(1).toSeq.last

  // println(s"w: $w b: $b")

  // sampling from posterior
  def proposal(wb : (Tensor1[Space], Tensor0)) : (Tensor1[Space], Tensor0) =
    val (w, b) = wb

    val nw = Normal(w, Tensor(0.1f, requiresGrad = false))
    val nb = Normal(b, Tensor(0.1f, requiresGrad = false))

    val neww = nw.sample()
    val newb = nb.sample()

    (neww, newb)

  
  val samples = MetropolisHastings.sample(
      f.tupled,
      proposal  , 
      (w0, b0)
    ).zipWithIndex.map { case (z, i) =>
      if i % 100 == 0 then
        println(s"Iteration $i: $z")
      z
    }
    .drop(1000)
    .take(1000)
    .toSeq
  
  val mean = samples.map(_._1).reduce(_.add(_)).mul(Tensor(1.0f / samples.size, requiresGrad = false))
  println("mean: " + mean)

@main def runBayesianLinearRegression() : Unit = 

    val w = 1f
    val b = 5f

    val xs = Seq.tabulate(100)(x => x.toFloat)
    val ys = xs.map(x => w * x + b)

    bayesianLinearRegression(xs, ys)
    println("done")