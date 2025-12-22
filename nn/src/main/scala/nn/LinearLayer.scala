package nn

import shapeful.*
import shapeful.tensor.Value
import shapeful.random.Random.Key

object LinearLayer:

  case class Params[In, Out, V](weight: Tensor2[In, Out, V], bias: Tensor1[Out, V])

  object Params:
    given [I: Label, O: Label, V]: TensorTree[Params[I, O, V]] = TensorTree.derived
    // given [I : Label, O : Label, V]: ToPyTree[Params[I, O, V]] = ToPyTree.derived

    def apply[In: Label, Out: Label, V: Value](paramKey: Key)(
        inputDim: Dim[In],
        outputDim: Dim[Out]
    ): Params[In, Out, V] = Params(
      weight = Tensor.of[V].randn(Shape(inputDim, outputDim), paramKey),
      bias = Tensor.of[V].zeros(Shape(outputDim))
    )

case class LinearLayer[In: Label, Out: Label, V: Value](params: LinearLayer.Params[In, Out, V])
    extends Function[Tensor1[In, V], Tensor1[Out, V]]:
  override def apply(x: Tensor1[In, V]): Tensor1[Out, V] =
    import params.{weight, bias}
    x.contract(Axis[In])(weight) + bias

object LinearMap:

  case class Params[In, V](weight: Tensor1[In, V], bias: Tensor0[V])

  object Params:
    given [In: Label, V]: TensorTree[Params[In, V]] = TensorTree.derived
    // given [In : Label, V]: ToPyTree[Params[In, V]] = ToPyTree.derived

    def apply[In: Label, V: Value](paramKey: Key)(inputDim: Dim[In]): Params[In, V] = Params(
      weight = Tensor.of[V].randn(Shape(inputDim), paramKey),
      bias = Tensor0(0.0f)(using summon[Value[V]])
    )

case class LinearMap[In: Label, V: Value](params: LinearMap.Params[In, V]) extends Function[Tensor1[In, V], Tensor0[V]]:
  override def apply(x: Tensor1[In, V]): Tensor0[V] =
    import params.{weight, bias}
    x.contract(Axis[In])(weight) + bias
