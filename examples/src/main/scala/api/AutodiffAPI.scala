package examples.api

import shapeful.*

@main
def autoDiffAPI(): Unit =
  val AB = FloatTensor
    .ones(
      Shape(
        Axis["A"] -> 10,
        Axis["B"] -> 5
      )
    )
  val AC = FloatTensor
    .ones(
      Shape(
        Axis["A"] -> 10,
        Axis["C"] -> 5
      )
    )
  val ABCD = FloatTensor
    .ones(
      Shape(
        Axis["A"] -> 2,
        Axis["B"] -> 3,
        Axis["C"] -> 4,
        Axis["D"] -> 5
      )
    )
  {
    def f(x: Tensor1["A", Float]): Tensor0[Float] = x.sum
    val df = Autodiff.grad(f)
    val delta = df(FloatTensor1.fromArray(Axis["A"], Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    type ParamsTuple = (Tensor2["A", "B", Float], Tensor1["C", Float])
    def f(params: ParamsTuple): Tensor0[Float] =
      params._1.sum + params._2.sum
    val df = Autodiff.grad(f)
    val delta = df(
      (
        FloatTensor2.fromArray(
          Axis["A"],
          Axis["B"],
          Array(
            Array.fill(5)(1.0f),
            Array.fill(5)(1.0f)
          )
        ),
        FloatTensor1.fromArray(Axis["C"], Array.fill(5)(1.0f))
      )
    )
    println((delta._1.shape, delta._2.shape))
  }
  {
    case class Params(
        a: Tensor2["A", "B", Float],
        b: Tensor1["C", Float]
    ) derives TensorTree
    def f(params: Params): Tensor0[Float] =
      params.a.sum + params.b.sum
    val df = Autodiff.grad(f)
    val delta = df(
      Params(
        FloatTensor2.fromArray(
          Axis["A"],
          Axis["B"],
          Array(
            Array.fill(5)(1.0f),
            Array.fill(5)(1.0f)
          )
        ),
        FloatTensor1.fromArray(Axis["C"], Array.fill(5)(1.0f))
      )
    )
    println(delta)
  }
  {
    def f(x: Tensor1["A", Float]): Tensor1["A", Float] = x
    val df = Autodiff.jacobian(f)
    val delta = df(FloatTensor1.fromArray(Axis["A"], Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    def f(x: Tensor1["A", Float]): Tensor2["A", "A", Float] = x.outerProduct(x)
    val df = Autodiff.jacobian(f)
    val delta = df(FloatTensor1.fromArray(Axis["A"], Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    import shapeful.tensor.TensorOps.*
    type ParamsTuple = (Tensor2["A", "B", Float], Tensor1["C", Float])
    def f(x: ParamsTuple): Tensor1["A", Float] = x._1.slice(Axis["B"] -> 0)
    val df = Autodiff.jacobian(f)
    val delta = df(
      (
        FloatTensor2.fromArray(
          Axis["A"],
          Axis["B"],
          Array(
            Array.fill(5)(1.0f),
            Array.fill(5)(1.0f)
          )
        ),
        FloatTensor1.fromArray(Axis["C"], Array.fill(5)(1.0f))
      )
    )
    println((delta._1.shape, delta._2.shape))
  }
  {
    println("Hessian")
    def f(x: Tensor1["A", Float]): Tensor0[Float] = x.sum
    val df = Autodiff.jacobian(f)
    val ddf = Autodiff.jacobian(df)
    val delta = ddf(FloatTensor1.fromArray(Axis["A"], Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    def f(x: Tensor1["A", Float]): Tensor0[Float] = x.sum
    val df = Autodiff.jacobian(f)
    val ddf = Autodiff.jacobian(df)
    val dddf = Autodiff.jacobian(ddf)
    val ddddf = Autodiff.jacobian(dddf)
    val delta = ddddf(FloatTensor1.fromArray(Axis["A"], Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    import shapeful.tensor.TensorOps.*
    type ParamsTuple = (Tensor2["A", "B", Float], Tensor1["C", Float])
    def f(x: ParamsTuple): Tensor0[Float] = x._1.sum
    val df = Autodiff.jacobian(f)
    val ddf = Autodiff.jacobian(df)
    val delta = ddf(
      (
        FloatTensor2.fromArray(
          Axis["A"],
          Axis["B"],
          Array(
            Array.fill(5)(1.0f),
            Array.fill(5)(1.0f)
          )
        ),
        FloatTensor1.fromArray(Axis["C"], Array.fill(5)(1.0f))
      )
    )
    // TODO Is this actually correct, check it!
    println(
      (
        (delta._1._1.shape, delta._1._2.shape),
        (delta._2._1.shape, delta._2._2.shape)
      )
    )
  }
