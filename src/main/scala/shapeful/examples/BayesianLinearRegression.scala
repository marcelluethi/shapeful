package shapeful.examples

// Example usage:

import shapeful.distributions.Normal
import shapeful.tensor.Tensor2
import shapeful.tensor.~>
import shapeful.tensor.Shape
import shapeful.tensor.Tensor1
import shapeful.tensor.Tensor0
import shapeful.tensor.Variable1
import shapeful.tensor.Variable0
import torch.Float32
import shapeful.tensor.TensorOps.*
import shapeful.tensor.Tensor1Ops.* 
import shapeful.tensor.Tensor2Ops.*
import shapeful.autodiff.Params
import shapeful.inference.MetropolisHastings

def bayesianLinearRegression(xs : Seq[Float], y : Seq[Float]) : Unit =

  val wShape = Shape("Features" ~> 1)
  val XShape = Shape("Data"~>xs.length, "Features" ~>1)
  val yShape = Shape("Data"~>y.length)

  val X = Tensor2.fromSeq(XShape, xs)
  val yt = Tensor1.fromSeq(yShape, y)

  val pw = Normal(Tensor1(wShape, 0f), Tensor1(wShape, 10f))
  val pb = Normal(Tensor0(1f), Tensor0(1f))


  def likelihood(w : Variable1["Features"], b : Variable0) : Tensor0[Float32] =
    val mu  = X.matmul1(w).add(b)
    val sigma = Tensor1(yShape, 1f)

    Normal(mu, sigma).logpdf(yt).mean

  def f(params : Params) : Tensor0[Float32] =

    val w = params.get[Variable1["Features"]]("w")
    val b = params.get[Variable0]("b")
    val logpw = pw.logpdf(w).sum
    val logpb = pb.logpdf(b)

    val ll = likelihood(w, b)
    logpw.add(logpb).add(ll)

  val params0 = new Params(
    Map("w" -> Variable1(wShape, 0.0f), 
    "b" -> Variable0(0.0f))
    )

  // sampling from posterior
  def proposal(param : Params) : Params =
    val w = param.get[Variable1["Features"]]("w")
    val b = param.get[Variable0]("b")

    val nw = Normal(w, Tensor1(w.shape, 0.1f))
    val nb = Normal(b, Tensor0(0.1f))

    val neww = nw.sample()
    val newb = nb.sample()

    param.update("w" -> neww.toVariable).update("b"-> newb.toVariable)

  val samples = MetropolisHastings.sample(
      f,
      proposal ,
      params0
    ).zipWithIndex.map { case (z, i) =>
      if i % 100 == 0 then
        println(s"Iteration $i: $z")
      z
    }
    .drop(1000)
    .take(1000)
    .toSeq

  val mean = samples.map(_.get[Variable1["Features"]]("w")).reduce((a, s) => a.add(s)).mul(Tensor0(1.0f / samples.size))
  println("mean: " + mean)

@main def runBayesianLinearRegression() : Unit =

    val w = 1f
    val b = 5f

    val xs = Seq.tabulate(100)(x => x.toFloat)
    val ys = xs.map(x => w * x + b)

    bayesianLinearRegression(xs, ys)
    println("done")
