package nn

import shapeful.*
import shapeful.random.Random
import shapeful.random.Random.Key

object LinearLayer:

    case class Params[In, Out](weight: Tensor2[In, Out, DType.Float32.type], bias: Tensor1[Out, DType.Float32.type])

    object Params:
        given [I : Label, O : Label]: TensorTree[Params[I, O]] = TensorTree.derived
        given [I : Label, O : Label]: ToPyTree[Params[I, O]] = ToPyTree.derived

        def apply[In : Label, Out : Label](paramKey: Key)(
            inputDim: Dim[In],
            outputDim: Dim[Out],
        ): Params[In, Out] = 
            val mean = Tensor0.of[DType.Float32.type].apply(0f)
            val std = Tensor0.of[DType.Float32.type].apply(1f)
            Params(
                weight = Random.normal(paramKey, Shape(inputDim, outputDim), mean, std),
                bias = Tensor.of[DType.Float32.type].zeros(Shape(outputDim)),
            )

case class LinearLayer[In : Label,Out : Label](params: LinearLayer.Params[In, Out]) extends Function[Tensor1[In, DType.Float32.type], Tensor1[Out, DType.Float32.type]]:
    override def apply(x: Tensor1[In, DType.Float32.type]): Tensor1[Out, DType.Float32.type] =
        import params.{weight, bias}
        x.contract(Axis[In])(weight) + bias

object LinearMap:

    case class Params[In](weight: Tensor1[In, Float32], bias: Tensor0[Float32])

    object Params:
        given [In : Label]: TensorTree[Params[In]] = TensorTree.derived
        given [In : Label]: ToPyTree[Params[In]] = ToPyTree.derived

        def apply[In : Label](paramKey: Key)(inputDim: Dim[In]): Params[In] = 
            val mean = Tensor0.of[Float32].apply(0f)
            val std = Tensor0.of[Float32].apply(1f)
            Params(
                weight = Random.normal(paramKey, Shape(inputDim), mean, std),
                bias = Tensor0.of[Float32].apply(0.0f),
            )

case class LinearMap[In : Label](params: LinearMap.Params[In]) extends Function[Tensor1[In, Float32], Tensor0[Float32]]:
    override def apply(x: Tensor1[In, Float32]): Tensor0[Float32] = 
        import params.{weight, bias}
        x.contract(Axis[In])(weight) + bias
