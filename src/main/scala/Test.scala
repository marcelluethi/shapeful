
// def bayes(xs : Seq[Float], y : Seq[Float]) : Unit =

//   type Data = "data"
//   given Dimension[Data] = Symbolic[Data](xs.size)

//   type Space = "space"
//   given Dimension[Space] = Symbolic[Space](1)

//   val X = Tensor.fromSeq[(Data, Space)](xs, requiresGrad = false)
//   val yt = Tensor.fromSeq[Tuple1[Data]](y, requiresGrad = false)


//   val pw = NormalDistribution(Tensor[Tuple1[Space]](0, requiresGrad = false), Tensor[Tuple1[Space]](1000, requiresGrad = false))


//   def likelihood(w : Tensor[Tuple1[Space]]) : Tensor[EmptyTuple.type] =
//     NormalDistribution[Data](X.dot(w), Tensor[Tuple1[Data]](1, requiresGrad = false)).logpdf(yt).mean[Data]

//   def f(w : Tensor[Tuple1[Space]]) : Tensor[EmptyTuple.type] =
//     val logpw = pw.logpdf(w).sum[Space]
//     val ll = likelihood(w)
//     logpw.add(ll)


//   val w = Tensor[Tuple1[Space]](1, requiresGrad = true)
//   val lr = 0.001f
//   val it = Iterator.iterate(w) { w =>
//     println("w: " + w)
//     val d = deriv(f)
//     val dw = d(w)

//     val neww = w.add(dw.multScalar(lr)).copy(requiresGrad = true)
//     neww
//   }
//   val z = it.take(50).toSeq.last
//   println(z)

// @main def test() : Unit =
//   // Create matrices with type-safety

// //  val w = 2.5f
// //  val b = 3f
// //
// //  val xs = Seq.tabulate(10)(x => x.toFloat)
// //  val ys = xs.map(x => w * x + b)
// //  println(xs)
// //  println(ys)
// //  //regression(xs, ys)
// //
// //  bayes(xs, ys)



// //  val x = Tensor[(3, 1)](1, requiresGrad = false)
// //  val y = Tensor[(1, 3)](1, requiresGrad = false)
// //
// //  x.mult(y)
// //  type Data = "data"
// //  given Dimension[Data] = Symbolic[Data](xs.size)
// //
// //  type Space = "space"
// //  given Dimension[Space] = Symbolic[Space](1)
// //  val t = Tensor[(Data, Space)](1, requiresGrad = true)
// //  //val ts : Tensor[Tuple1[Data]] = t.sumT[Space]

// //  val m1 = Tensor[((2,2), (2, 2))](List(2, 2), 0.0)
// //  val m2 = Tensor[((2,2), (2, 2))](List(2, 2), 0.0)
// //  val m3 = Tensor.add(Tensor.mult(m1, m2), m2)

// // Type-safe matrix multiplication (compiles because dimensions match)
// // val result = Tensor.mult(m1, m2) // Result type is Tensor[(2, 2), Double]

// // This would not compile:
// // val wrongResult = Tensor.matMul(m1, m1) // Dimension mismatch error

//   type Data = "data"
//   given Dimension[Data] = Symbolic[Data](2)

//   type Space = "space"
//   given Dimension[Space] = Symbolic[Space](2)

//   type Param = "param"

//   given Dimension[Param] = Symbolic[Param](2)


//   val tt = Tensor.fromSeq[(Data, Space, Param)]( Seq(1,2, 3, 4, 5, 6,7,8),requiresGrad = true)
//   println("tt " + tt)
//   val x : Tensor[(Space, Param)] = tt.apply[Data](0)
//   println(x)

// //  println("sums \n=======")
// //  println(tt.sum[Space])