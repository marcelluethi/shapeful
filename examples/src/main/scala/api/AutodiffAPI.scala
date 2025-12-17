package examples.api

import shapeful.* 

@main
def autoDiffAPI(): Unit =
  val AB = Tensor.ones(Shape(
    Axis["A"] -> 10,
    Axis["B"] -> 5,
  ))
  val AC = Tensor.ones(Shape(
      Axis["A"]-> 10,
      Axis["C"] -> 5
  ))
  val ABCD = Tensor.ones(Shape(
      Axis["A"] -> 2,
      Axis["B"] -> 3,
      Axis["C"] -> 4,
      Axis["D"] -> 5,
  ))
  {
    def f(x: Tensor1["A"]): Tensor0 = x.sum
    val df = Autodiff.grad(f)
    val delta = df(Tensor1(Axis["A"], Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    type ParamsTuple = (Tensor2["A", "B"], Tensor1["C"])
    def f(params: ParamsTuple): Tensor0 =
      params._1.sum + params._2.sum
    val df = Autodiff.grad(f)
    val delta = df((
      Tensor2(Axis["A"], Axis["B"], Array(
        Array.fill(5)(1.0f),
        Array.fill(5)(1.0f),
      )),
      Tensor1(Axis["C"], Array.fill(5)(1.0f))
    ))
    println((delta._1.shape, delta._2.shape))
  }
  {
    case class Params(
      a: Tensor2["A", "B"], 
      b: Tensor1["C"],
    ) derives TensorTree
    def f(params: Params): Tensor0 =
      params.a.sum + params.b.sum
    val df = Autodiff.grad(f)
    val delta = df(Params(
      Tensor2(Axis["A"], Axis["B"], Array(
        Array.fill(5)(1.0f),
        Array.fill(5)(1.0f),
      )),
      Tensor1(Axis["C"], Array.fill(5)(1.0f))
    ))
    println(delta)
  }
  {
    def f(x: Tensor1["A"]): Tensor1["A"] = x
    val df = Autodiff.jacobian(f)
    val delta = df(Tensor1(Axis["A"], Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    def f(x: Tensor1["A"]): Tensor2["A", "A"] = x.outerProduct(x)
    val df = Autodiff.jacobian(f)
    val delta = df(Tensor1(Axis["A"], Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    import shapeful.tensor.TensorOps.*
    type ParamsTuple = (Tensor2["A", "B"], Tensor1["C"])
    def f(x: ParamsTuple): Tensor1["A"] = x._1.slice(Axis["B"] -> 0)
    val df = Autodiff.jacobian(f)
    val delta = df((
      Tensor2(Axis["A"], Axis["B"], Array(
        Array.fill(5)(1.0f),
        Array.fill(5)(1.0f),
      )),
      Tensor1(Axis["C"], Array.fill(5)(1.0f))
    ))
    println((delta._1.shape, delta._2.shape))
  }
  {
    println("Hessian")
    def f(x: Tensor1["A"]): Tensor0 = x.sum
    val df = Autodiff.jacobian(f)
    val ddf = Autodiff.jacobian(df)
    val delta = ddf(Tensor1(Axis["A"], Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    def f(x: Tensor1["A"]): Tensor0 = x.sum
    val df = Autodiff.jacobian(f)
    val ddf = Autodiff.jacobian(df)
    val dddf = Autodiff.jacobian(ddf)
    val ddddf = Autodiff.jacobian(dddf)
    val delta = ddddf(Tensor1(Axis["A"], Array.fill(10)(1.0f)))
    println(delta.shape)
  }
  {
    import shapeful.tensor.TensorOps.*
    type ParamsTuple = (Tensor2["A", "B"], Tensor1["C"])
    def f(x: ParamsTuple): Tensor0 = x._1.sum
    val df = Autodiff.jacobian(f)
    val ddf = Autodiff.jacobian(df)
    val delta = ddf((
      Tensor2(Axis["A"], Axis["B"], Array(
        Array.fill(5)(1.0f),
        Array.fill(5)(1.0f),
      )),
      Tensor1(Axis["C"], Array.fill(5)(1.0f))
    ))
    // TODO Is this actually correct, check it!
    println((
      (delta._1._1.shape, delta._1._2.shape),
      (delta._2._1.shape, delta._2._2.shape)
    ))
  }