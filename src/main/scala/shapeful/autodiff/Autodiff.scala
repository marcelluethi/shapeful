package shapeful.autodiff

import scala.language.experimental.namedTypeArguments
import shapeful.*
import me.shadaj.scalapy.py

/**
 * Typeclass defining the relationship between a parameter type P and its gradient type G.
 * This allows for type-safe gradient computation where gradients are explicitly typed.
 */
trait GradType[P, G]:
  /** Convert parameter and computed gradient value to the gradient type */
  def toGradient(params: P, gradientValue: P): G
  
  /** Convert from gradient type back to parameter type if needed */
  def fromGradient(grad: G): P



/** Companion object with helper methods and default instances */
object GradType:
  /** Access an implicit GradType instance */
  def apply[P, G](using gt: GradType[P, G]): GradType[P, G] = gt
  
  /** Default instance where parameters and gradients have the same type */
  given identityGradType[P]: GradType[P, P] = new GradType[P, P]:
    def toGradient(params: P, gradientValue: P): P = gradientValue
    def fromGradient(grad: P): P = grad
      

object Autodiff:

  def grad[Params: ToPyTree, Gradient](f: Params => Tensor[EmptyTuple])(using gradType : GradType[Params, Gradient]): Params => Gradient =

    val ptree = summon[ToPyTree[Params]]

    val fpy = (jxpr: py.Dynamic) =>
      val x = summon[ToPyTree[Params]].fromPyTree(jxpr)
      f(x).jaxValue

    val gpy = Jax.jax_helper.grad(fpy)

    (params: Params) =>
      val xpy = ptree.toPyTree(params)
      val pygrad = gpy(xpy) // Evaluate the function expression
      gradType.toGradient(params, ptree.fromPyTree(pygrad)) // Convert back to Params

  def valueAndGrad[Params: ToPyTree](f: Params => Tensor[EmptyTuple]): Params => (Tensor[EmptyTuple], Params) =

    val ptree = summon[ToPyTree[Params]]

    val fpy = (jxpr: py.Dynamic) =>
      val x = summon[ToPyTree[Params]].fromPyTree(jxpr)
      f(x).jaxValue

    val vgpy = Jax.jax_helper.value_and_grad(fpy)

    (params: Params) =>
      val xpy = ptree.toPyTree(params)
      val result = vgpy(xpy) // Returns tuple of (value, grad)
      val pyvalue = result.__getitem__(0).as[Jax.PyDynamic]
      val pygrad = result.__getitem__(1).as[Jax.PyAny]
      val value = new Tensor[EmptyTuple](Shape.empty, pyvalue)
      val grad = ptree.fromPyTree(pygrad)
      (value, grad)


  def jacFwd[InputDim <: Label, OutputDim <: Label](
      f: Tensor1[InputDim] => Tensor1[OutputDim]
  ): Tensor1[InputDim] => Tensor2[OutputDim, InputDim] =

    val inputToPyTree = summon[ToPyTree[Tensor1[InputDim]]]

    val fpy = (jxpr: py.Dynamic) =>
      val x = inputToPyTree.fromPyTree(jxpr)
      f(x).jaxValue

    val jfpy = Jax.jax_helper.jacfwd(fpy)

    (params: Tensor1[InputDim]) =>
      val xpy = inputToPyTree.toPyTree(params)
      val pyjac = jfpy(xpy) // Returns Jacobian matrix
      // Create 2D shape for Jacobian matrix
      val outputDim = f(params).shape.dims(0) // m (number of outputs)
      val inputDim = params.shape.dims(0) // n (number of inputs)
      val jacobianShape = Shape2[OutputDim, InputDim](outputDim, inputDim)

      new Tensor2[OutputDim, InputDim](jacobianShape, pyjac.as[Jax.PyDynamic], params.dtype)

  def jacRev[InputDim <: Label, OutputDim <: Label](
      f: Tensor1[InputDim] => Tensor1[OutputDim]
  ): Tensor1[InputDim] => Tensor2[OutputDim, InputDim] =

    val inputToPyTree = summon[ToPyTree[Tensor1[InputDim]]]

    val fpy = (jxpr: py.Dynamic) =>
      val x = inputToPyTree.fromPyTree(jxpr)
      f(x).jaxValue

    val jrpy = Jax.jax_helper.jacrev(fpy)

    (params: Tensor1[InputDim]) =>
      val xpy = inputToPyTree.toPyTree(params)
      val pyjac = jrpy(xpy) // Returns Jacobian matrix

      // Create 2D shape for Jacobian matrix
      val outputDim = f(params).shape.dims(0) // m (number of outputs)
      val inputDim = params.shape.dims(0) // n (number of inputs)
      val jacobianShape = Shape2[OutputDim, InputDim](outputDim, inputDim)

      new Tensor2[OutputDim, InputDim](jacobianShape, pyjac.as[Jax.PyDynamic], params.dtype)
