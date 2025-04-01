// package shapeful.tensor

// import munit.FunSuite
// import shapeful.tensor.Tensor.Tensor0
// import shapeful.tensor.Tensor
// import shapeful.tensor.TensorOps.*
// import shapeful.tensor.Tensor0Ops.*
// import shapeful.tensor.Tensor.Tensor1
// import shapeful.nn.Softmax

// class ActivationTets extends FunSuite {

//   test("softmax normalizes correctly") {

//     val tensor = Tensor.fromSeq(Shape["Data", "Output"](2, 3), List(1f, 2f, 3f, 4f, 5f, 6f))
//     val softmax = new Softmax["Output"]()
//     val smtensor = softmax(tensor).sum["Output"]

//     for (i <- 0 until smtensor.dim["Data"]) {
//       assertEqualsFloat(smtensor.apply["Data"](i).item, 1.0, 0.01)
//     }
//   }

// }
