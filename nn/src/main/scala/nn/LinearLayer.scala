package nn

import shapeful.*
import shapeful.random.Random.Key

object LinearLayer:

  case class Params[In, Out, V](weight: Tensor2[In, Out, V], bias: Tensor1[Out, V])

  object Params:
    given [I: Label, O: Label, V]: TensorTree[Params[I, O, V]] = TensorTree.derived
    // given [I : Label, O : Label, V]: ToPyTree[Params[I, O, V]] = ToPyTree.derived

    def apply[In: Label, Out: Label, V: ScalarValue](paramKey: Key)(
        inputDim: Dim[In],
        outputDim: Dim[Out]
    ): Params[In, Out, V] = Params(
      weight = Tensor(TensorValue[V]).randn(Shape(inputDim, outputDim))(paramKey),
      bias = Tensor(TensorValue[V]).zeros(Shape(outputDim))
    )

case class LinearLayer[In: Label, Out: Label, V](params: LinearLayer.Params[In, Out, V])
    extends Function[Tensor1[In, V], Tensor1[Out, V]]:
  override def apply(x: Tensor1[In, V]): Tensor1[Out, V] =
    import params.{weight, bias}
    x.contract(Axis[In])(weight) + bias

object LinearMap:

  case class Params[In](weight: Tensor1[In, Float], bias: Tensor0[Float])

  object Params:
    given [In: Label]: TensorTree[Params[In]] = TensorTree.derived
    // given [In : Label, V]: ToPyTree[Params[In, V]] = ToPyTree.derived

    def apply[In: Label](paramKey: Key)(inputDim: Dim[In]): Params[In] = Params(
      weight = FloatTensor.randn(Shape(inputDim))(paramKey),
      bias = FloatTensor0.zero
    )

case class LinearMap[In: Label, V](params: LinearMap.Params[In]) extends (FloatTensor1[In] => FloatTensor0):
  override def apply(x: FloatTensor1[In]): FloatTensor0 =
    import params.{weight, bias}
    x.contract(Axis[In])(weight) + bias
