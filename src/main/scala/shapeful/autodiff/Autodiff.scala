package shapeful.autodiff

import scala.language.experimental.namedTypeArguments
import shapeful.*
import me.shadaj.scalapy.py

object Autodiff:

  def grad[Params: ToPyTree](f: Params => Tensor[EmptyTuple]): Params => Params =

    val ptree = summon[ToPyTree[Params]]

    val fpy = (jxpr: py.Dynamic) =>
      val x = summon[ToPyTree[Params]].fromPyTree(jxpr)
      f(x).jaxValue

    val gpy = Jax.jax_helper.grad(fpy)

    (params: Params) =>
      val xpy = ptree.toPyTree(params)
      val pygrad = gpy(xpy) // Evaluate the function expression
      ptree.fromPyTree(pygrad) // Convert back to Params
