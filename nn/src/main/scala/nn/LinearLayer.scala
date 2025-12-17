package nn

import shapeful.*
import shapeful.Conversions.given
import shapeful.random.Random.Key

object LinearLayer:

    case class Params[In, Out](weight: Tensor2[In, Out], bias: Tensor1[Out])

    object Params:
        given [I : Label, O : Label]: TensorTree[Params[I, O]] = TensorTree.derived
        given [I : Label, O : Label]: ToPyTree[Params[I, O]] = ToPyTree.derived

        def apply[In : Label, Out : Label](paramKey: Key)(
            inputDim: Dim[In],
            outputDim: Dim[Out],
        ): Params[In, Out] = Params(
            weight = Tensor.randn(Shape(inputDim, outputDim), paramKey),
            bias = Tensor.zeros(Shape(outputDim)),
        )

case class LinearLayer[In : Label,Out : Label](params: LinearLayer.Params[In, Out]) extends Function[Tensor1[In], Tensor1[Out]]:
    override def apply(x: Tensor1[In]): Tensor1[Out] =
        import params.{weight, bias}
        x.contract(Axis[In])(weight) + bias

object LinearMap:

    case class Params[In](weight: Tensor1[In], bias: Tensor0)

    object Params:
        given [In : Label]: TensorTree[Params[In]] = TensorTree.derived
        given [In : Label]: ToPyTree[Params[In]] = ToPyTree.derived

        def apply[In : Label](paramKey: Key)(inputDim: Dim[In]): Params[In] = Params(
            weight = Tensor.randn(Shape(inputDim), paramKey),
            bias = Tensor0(0.0f),
        )

case class LinearMap[In : Label](params: LinearMap.Params[In]) extends Function[Tensor1[In], Tensor0]:
    override def apply(x: Tensor1[In]): Tensor0 = 
        import params.{weight, bias}
        x.contract(Axis[In])(weight) + bias
