package shapeful.examples


// Example usage:

import shapeful.distributions.NormalDistribution
import shapeful.tensor.{Dimension, Tensor, addScalar, cov, dot, mult, multScalar, pow, sub, add}
import shapeful.tensor.Dimension.Symbolic
import shapeful.tensor.Tensor.{Tensor0, Tensor1, Tensor2}
import shapeful.autodiff.DifferentiableFunction2
import shapeful.autodiff.GeneralGradientDescent
import shapeful.autodiff.TensorTupleOps


def bayesianLinearRegression(xs : Seq[Float], y : Seq[Float]) : Unit =

  type Data = "data"
  given Dimension[Data] = Symbolic[Data](xs.size)

  type Space = "space"
  given Dimension[Space] = Symbolic[Space](1)

  given TensorTupleOps[(Tensor[Tuple1[Space]], Tensor[EmptyTuple.type])] with
     def update(
        params: (Tensor[Tuple1[Space]], Tensor[EmptyTuple.type]), 
        gradients: (Tensor[Tuple1[Space]], Tensor[EmptyTuple.type]), 
        lr: Tensor[EmptyTuple.type]): (Tensor[Tuple1[Space]], Tensor[EmptyTuple.type]) = 
            val (x, y) = params
            val (dx, dy) = gradients
            (x.add(dx.multScalar(lr)).copy(requiresGrad = true), y.add(dy.multScalar(lr)).copy(requiresGrad = true))



  val X = Tensor.fromSeq[(Data, Space)](xs, requiresGrad = false)
  val yt = Tensor.fromSeq[Tuple1[Data]](y, requiresGrad = false)

  val pw = NormalDistribution[Tuple1[Space]](Tensor(1f, requiresGrad = false), Tensor(10f, requiresGrad = false))
  val pb = NormalDistribution[EmptyTuple](Tensor(3f, requiresGrad = false), Tensor(10f, requiresGrad = false))

  def likelihood(w : Tensor1[Space]) : Tensor0 =
    val mu : Tensor1[Data] = X.dot(w)
    val sigma : Tensor1[Data] = Tensor(1, requiresGrad = false)

    NormalDistribution[Tuple1[Data]](mu, sigma).logpdf(yt).mean[Data]

  def f(w : Tensor1[Space], b: Tensor0) : Tensor0 =
    val logpw = pw.logpdf(w).sum[Space]
    val logpb = pb.logpdf(b).sum[Space]

    val ll = likelihood(w)
    logpw.add(logpb).add(ll)


  // optimize
  val lr : Tensor0 = Tensor(0.01f, requiresGrad = false)
  val w0 : Tensor1[Space] = Tensor(0.0, requiresGrad = true)
  val b0 : Tensor0 = Tensor(2.0, requiresGrad = true)

  val diffFun = DifferentiableFunction2(f)
  val gd = GeneralGradientDescent(0.01, 100)
  val (w, b) = gd.optimize(diffFun, (w0, b0))

//   val (w, b) = (0 until 1000).foldLeft((w0, b0)) { case ((w, b), i) =>
//     if i % 100 == 0 then
//         println(s"Iteration $i: $w $b")
  
//     val (wg, wb) = new DifferentiableFunction(f).deriv(w, b)

//     val neww = w.add(wg.multScalar(lr)).copy(requiresGrad = true)
//     val newb = b.add(wb.multScalar(lr)).copy(requiresGrad = true)
  
//     (neww, newb)
//   }
//   val z = (w, b)

  println(s"w: $w b: $b")

@main def runBayesianLinearRegression() : Unit = 
      // Create matrices with type-safety

    val w = 2.5f
    val b = 3f

    val xs = Seq.tabulate(10)(x => x.toFloat)
    val ys = xs.map(x => w * x + b)

    bayesianLinearRegression(xs, ys)
    println("done")