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
import shapeful.tensor.Shape


def bayesianLinearRegression(xs : Seq[Float], y : Seq[Float]) : Unit =

  type Data = "data"
  type Space = "space"
  val spaceShape = Shape[Space](1)
  val dataShape = Shape[Data](xs.size)
  

  val X = Tensor.fromSeq(dataShape ++ spaceShape, xs)
  val yt = Tensor.fromSeq(dataShape, y)

  val pw = Normal(spaceShape, Tensor(spaceShape, 0f), Tensor(spaceShape, 10f))
  val pb = Normal(Shape.empty, Tensor(Shape.empty, 1), Tensor(Shape.empty, 100f))

  def likelihood(w : Tensor1[Space], b : Tensor0) : Tensor0 =
    val mu : Tensor1[Data] = X.matmul(w).add(b)
    val sigma : Tensor1[Data] = Tensor(dataShape, 1)

    Normal[Tuple1[Data]](dataShape, mu, sigma).logpdf(yt).mean[Data]

  def f(w : Tensor1[Space], b: Tensor0) : Tensor0 =
    val logpw = pw.logpdf(w).sum[Space]
    val logpb = pb.logpdf(b).sum[Space]

    val ll = likelihood(w, b)
    logpw.add(logpb).add(ll)



  val w0 : Tensor1[Space] = Tensor(spaceShape, 0.0, requiresGrad = true)
  val b0 : Tensor0 = Tensor(Shape.empty, 0.0, requiresGrad = true)


  // sampling from posterior
  def proposal(wb : (Tensor1[Space], Tensor0)) : (Tensor1[Space], Tensor0) =
    val (w, b) = wb

    val nw = Normal(w.shape, w, Tensor(w.shape, 0.1f, requiresGrad = false))
    val nb = Normal(b.shape, b, Tensor(b.shape, 0.1f, requiresGrad = false))

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
  
  val mean = samples.map(_._1).reduce(_.add(_)).mul(Tensor(Shape.empty, 1.0f / samples.size))
  println("mean: " + mean)

@main def runBayesianLinearRegression() : Unit = 

    val w = 1f
    val b = 5f

    val xs = Seq.tabulate(100)(x => x.toFloat)
    val ys = xs.map(x => w * x + b)

    bayesianLinearRegression(xs, ys)
    println("done")