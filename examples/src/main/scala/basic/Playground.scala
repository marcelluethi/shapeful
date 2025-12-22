// package examples.basic

// import shapeful.*
// import shapeful.tensor.Value.Float32

// @main def playground(): Unit =
//   println("TensorV2 Playground")
//   {
//     println("MatMul tests")
//     val values = Array(
//       0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
//       1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
//     ).map(_.toFloat)
//     val X = Tensor2["Samples", "Features", Float32](
//       values = values,
//       shape = Shape(
//         Axis["Samples"] -> 10,
//         Axis["Features"] -> 2,
//       )
//     )
//     val XT = X.transpose
//     val XTX = XT.matmul(X)
//     val XXT = X.matmul(XT)
//     println(XTX.shape)
//     println(XXT.shape)
//   }
//   {
//     println("Normalization example")
//     val values = Array(
//       0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
//       1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
//     ).map(_.toFloat)
//     val X = Tensor2["Samples", "Features", Float32](
//       values = values,
//       shape = Shape(
//         Axis["Samples"] -> 10,
//         Axis["Features"] -> 2,
//       )
//     )
//     val means = X.vmap(Axis["Features"])(_.mean)
//     val stds = X.vmap(Axis["Features"])(_.std)
//     val Xnorm = X.vmap(Axis["Samples"]){ (x) =>
//       (x - means) / stds
//     }
//     println(Xnorm)
//     println(Xnorm.shape)
//     println(Xnorm.device)
//     println(Xnorm.dtype)
//   }
//   {
//     println("DType and Device tests")
//     val t = Tensor.zeros(Shape(
//       Axis["Batch"] -> 1024,
//       Axis["Features"] -> 512,
//     ))
//     println(t.shape)
//     println(t.dtype)
//     println(t.asType(DType.Int32).dtype)
//     println(t.device)
//     println(t.toDevice(Device.CPU).device)
//   }
//   {
//     val x = Tensor.zeros(Shape(
//       Axis["Features"] -> 2,
//     ))
//     val A = Tensor.zeros(Shape(
//       Axis["Samples"] -> 50,
//       Axis["Features"] -> 2,
//     ))
//     // val y1 = x.contract(Axis["A"])(A)
//     val y1 = A.contract(Axis["Features"])(x)
//     println(y1.shape)
//     // A.contract(Axis["lala"])(x)
//     // A.contract(Axis["Samples"])(x)
//     val y2 = x.contract(Axis["Features"])(A)
//     println(y2.shape)
//     val y3 = x.outerProduct(A)
//     println(y3.shape)
//   }
//   {
//     println("Einops rearrange tests")
//     type Batch = "batch"
//     type Frame = "frame"
//     type BatchFrame = "batch_frame"
//     type Width = "width"
//     type Height = "height"
//     type Channel = "channel"
//     val X = Tensor.zeros(Shape(
//       Axis[Batch] -> 32,
//       Axis[Frame] -> 64,
//       Axis[Width] -> 256,
//       Axis[Height] -> 256,
//       Axis[Channel] -> 3,
//     ))
//     val d = X.rearrange(
//       (
//         Axis[Batch * Frame],
//         Axis[Width * Height],
//         Axis[Channel]
//       )
//     )
//     println(d.shape)
//     val e = d.as[(Axis[Frame], Axis["pixel"], Axis[Channel])]
//     println(e.shape)
//   }
//   {
//     println("Contraction with overlapping axes")
//     import scala.util.NotGiven
//     def f[L1: Label, L2: Label, L3: Label](
//       x: Tensor[(L1, L2)],
//       y: Tensor[(L2, L3)]
//     ): Tensor[(L1, L3, L2)] =
//       x.vmap(Axis[L1]){ xi =>
//         y.vmap(Axis[L3]){ yi =>
//           xi + yi
//         }
//       }
//     val z = f(Tensor.zeros(Shape(
//       Axis["A"] -> 2,
//       Axis["B"] -> 3,
//     )), Tensor.zeros(Shape(
//       Axis["B"] -> 3,
//       Axis["C"] -> 4,
//     )))
//     println(z.shape)
//   }
//   {
//     def f(t1: Tensor[("A", "C")], t2: Tensor[Tuple1["C"]]): Tensor[Tuple1["A"]] =
//       t1.matmul(t2)
//     val t1 = Tensor.ones(Shape(
//       Axis["A"] -> 2,
//       Axis["C"] -> 2,
//     ))
//     val t2 = Tensor.ones(Shape(
//       Axis["C"] -> 2,
//     ))
//     println(f(t1, t2))
//     println("vmap 2")
//     import scala.util.NotGiven
//     val x1 = Tensor.ones(Shape(
//       Axis["B"] -> 1,
//       Axis["A"] -> 2,
//       Axis["C"] -> 2,
//     ))
//     val x2 = Tensor.ones(Shape(
//       Axis["B"] -> 1,
//       Axis["C"] -> 2,
//     ))
//   }
//   {
//     def f[L1: Label, L2: Label, L3: Label](x: Tensor[(L1, L2)], y: Tensor[(L2, L3)]) =
//       x.vmap(Axis[L1]){ xi =>
//         y.vmap(Axis[L3]){ yi =>
//           xi + yi
//         }
//       }
//     println(f(
//       Tensor.zeros(Shape(
//         Axis["A"] -> 2,
//         Axis["B"] -> 3,
//       )),
//       Tensor.zeros(Shape(
//         Axis["B"] -> 3,
//         Axis["C"] -> 4,
//       ))
//     ).shape)
//   }

//   {
//     println("Ravel")
//     val res = Tensor.ones(Shape(
//       Axis["A"] -> 2,
//       Axis["B"] -> 3,
//       Axis["C"] -> 4,
//     )).ravel
//     println(res.shape)
//   }
//   {
//     println("swapaxes")
//     val res = Tensor.ones(Shape(
//       Axis["A"] -> 2,
//       Axis["B"] -> 3,
//       Axis["C"] -> 4,
//     )).swap(Axis["A"], Axis["C"])
//     println(res.shape)
//   }
//   {
//     println("appendAxis / prependAxis")
//     val res = Tensor.ones(Shape(
//       Axis["A"] -> 2,
//       Axis["B"] -> 3,
//       Axis["C"] -> 4,
//     )).appendAxis(Axis["D"])
//     println(res.shape)
//     val res2 = Tensor.ones(Shape(
//       Axis["A"] -> 2,
//       Axis["B"] -> 3,
//       Axis["C"] -> 4,
//     )).prependAxis(Axis["D"])
//     println(res2.shape)
//   }
//   {
//     println("squeeze")
//     val res = Tensor.ones(Shape(
//       Axis["A"] -> 1,
//       Axis["B"] -> 3,
//       Axis["C"] -> 1,
//     )).squeeze(Axis["A"])
//     println(res.shape)
//     val res2 = res.squeeze(Axis["C"])
//     println(res2.shape)
//   }
//   {
//     println("Slice")
//     val res = Tensor.ones(Shape(
//       Axis["A"] -> 2,
//       Axis["B"] -> 3,
//     )).slice(
//       Axis["B"] -> 2
//     )
//     println(res.shape)
//     val res2 = Tensor.ones(Shape(
//       Axis["A"] -> 2,
//       Axis["B"] -> 3,
//     )).slice(
//       Axis["B"] -> (0 to 1)
//     )
//     println(res2.shape)
//     val res3 = Tensor.ones(Shape(
//       Axis["A"] -> 2,
//       Axis["B"] -> 3,
//       Axis["C"] -> 4,
//       Axis["D"] -> 5,
//     )).slice((
//       Axis["B"] -> 2,
//       Axis["C"] -> 3,
//     ))
//     println(res3.shape)
//   }
//   {
//     println("zipvmap tests")
//     type Batch = "Batch"
//     type Asset = "Asset"
//     type Region = "Region"
//     type Sector = "Sector"
//     type Risk = "Risk"

//     val x = Tensor.ones(Shape(
//       Axis[Batch] -> 6,
//       Axis[Asset] -> 3,
//       Axis[Region] -> 5,
//     ))

//     val y = Tensor.ones(Shape(
//       Axis[Region] -> 5,
//       Axis[Batch] -> 6,
//       Axis[Sector] -> 4,
//     ))

//     val z = Tensor.ones(Shape(
//       Axis[Sector] -> 4,
//       Axis[Risk] -> 5,
//       Axis[Batch] -> 6,
//     ))

//     val res = zipvmap(Axis[Batch])(x, y) {
//       case (xi, yi) => xi.sum + yi.sum
//     }
//     println(res.shape)

//     val res2 = zipvmap(Axis[Batch])(x, y, z) {
//       (xi, yi, zi) => xi.sum + yi.sum + zi.sum
//     }
//     println(res2.shape)
//   }
//   {
//     import shapeful.tensor.* // Assuming imports

//     type Batch = "Batch"
//     type Asset = "Asset"
//     type Region = "Region"
//     type Sector = "Sector"
//     type Risk = "Risk"

//     val x = Tensor.ones(Shape(Axis[Batch] -> 6, Axis[Asset] -> 3, Axis[Region] -> 5))
//     val y = Tensor.ones(Shape(Axis[Region] -> 5, Axis[Batch] -> 6, Axis[Sector] -> 4))
//     val z = Tensor.ones(Shape(Axis[Sector] -> 4, Axis[Risk] -> 5, Axis[Batch] -> 6))

//     val res = zipvmap(Axis[Batch])((x, y, z)) {
//       case (xi, yi, zi) => xi.sum + yi.sum + zi.sum
//     }
//     println(res.shape)
//   }
//   {
//     println("TensorWhere tests")
//     val x = Tensor.ones(Shape(
//       Axis["A"] -> 2,
//       Axis["B"] -> 3,
//     ))
//     val y = Tensor.zeros(Shape(
//       Axis["A"] -> 2,
//       Axis["B"] -> 3,
//     ))
//     val condition = Tensor.zeros(Shape(
//       Axis["A"] -> 2,
//       Axis["B"] -> 3,
//     )).asType(DType.Bool)
//     val res = where(condition, x, y)
//     println(res.shape)
//   }
//   {
//     println("Diag")
//     val x = Tensor.ones(Shape(
//       Axis["A"] -> 2,
//       Axis["B"] -> 3,
//     ))
//     val res = x.diagonal
//     println(res.shape)
//   }
//   {
//     println("Set")
//     val x = Tensor.ones(Shape(
//       Axis["A"] -> 2,
//       Axis["B"] -> 3,
//     )).set((
//       Axis["A"] -> 1,
//       Axis["B"] -> 2,
//     ))(Tensor0(42))
//     println(x)
//     val v = Tensor1(
//       Axis["B"],
//       Array(100, 101, 102),
//     )
//     val x2 = Tensor.ones(Shape(
//       Axis["A"] -> 2,
//       Axis["B"] -> 3,
//     )).set(
//       Axis["A"] -> 1,
//     )(v)
//     println(x2)
//   }
//   {
//     // attention mechanism example
//     def softmax[L: Label](tensor: Tensor1[L]): Tensor1[L] =
//       val expTensor = tensor.exp
//       val sumExp = expTensor.sum
//       expTensor.vmap(Axis[L]) { _ / sumExp }

//     val X = Tensor.ones(
//       Shape(Axis["Batch"] -> 32, Axis["Sequence"] -> 128, Axis["Value"] -> 64)
//     )
//     val WK = Tensor.ones(
//       Shape(Axis["Value"] -> 64, Axis["Key"] -> 64)
//     )
//     val WQ = Tensor.ones(
//       Shape(Axis["Value"] -> 64, Axis["Query"] -> 64)
//     )
//     val WV = Tensor.ones(
//       Shape(Axis["Value"] -> 64, Axis["NewValue"] -> 64)
//     )
//     val Xnew = X.vmap(Axis["Batch"]) { Xi =>
//       val K = Xi.contract(Axis["Value"])(WK)
//       val Q = Xi.contract(Axis["Value"])(WQ)
//       val V = Xi.contract(Axis["Value"])(WV)
//       val dk = Tensor0(Math.sqrt(K.shape(Axis["Key"])).toFloat)
//       val AttnWeights = (Q.contract(Axis["Query"] -> Axis["Key"])(K) :/ dk)
//         .as[(Axis["Sequence"], Axis["Weights"])]
//         .vmap(Axis["Sequence"])(softmax)
//       val res = AttnWeights.contract(Axis["Weights"] -> Axis["Sequence"])(V)
//       res.relabel(Axis["NewValue"] -> Axis["Value"])
//     }
//     println(Xnew.shape)
//   }
//   {
//     // multi-head attention mechanism example
//     def softmax[L: Label](tensor: Tensor1[L]): Tensor1[L] =
//       val expTensor = tensor.exp
//       val sumExp = expTensor.sum
//       expTensor.vmap(Axis[L]) { _ / sumExp }

//     val X = Tensor.ones(
//       Shape(Axis["Batch"] -> 32, Axis["Sequence"] -> 128, Axis["Value"] -> 64)
//     )
//     val WK = Tensor.ones(
//       Shape(Axis["Value"] -> 64, Axis["Key"] -> 8, Axis["Heads"] -> 8)
//     )
//     val WQ = Tensor.ones(
//       Shape(Axis["Value"] -> 64, Axis["Query"] -> 8, Axis["Heads"] -> 8)
//     )
//     val WV = Tensor.ones(
//       Shape(Axis["Value"] -> 64, Axis["NewValue"] -> 8, Axis["Heads"] -> 8)
//     )
//     val Xnew = X.vmap(Axis["Batch"]) { Xi =>
//       val K = Xi.contract(Axis["Value"])(WK)
//       val Q = Xi.contract(Axis["Value"])(WQ)
//       val V = Xi.contract(Axis["Value"])(WV)
//       val res = zipvmap(Axis["Heads"])(Q, K, V) { (Qi, Ki, Vi) =>
//         val dk = Tensor0(Math.sqrt(Ki.shape(Axis["Key"])).toFloat)
//         val AttnWeights = (Qi.contract(Axis["Query"] -> Axis["Key"])(Ki) :/ dk)
//           .as[(Axis["Sequence"], Axis["Weights"])]
//           .vmap(Axis["Sequence"])(softmax)
//         AttnWeights.contract(Axis["Weights"] -> Axis["Sequence"])(Vi)
//       }
//       res.rearrange((
//         Axis["Sequence"],
//         Axis["Heads" * "NewValue"],
//       )).relabel(Axis["Heads" * "NewValue"] -> Axis["Value"])
//     }
//     println(Xnew.shape)
//   }
